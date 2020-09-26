from data import *
from net import *
from lib import *
from eval import *
import random
import datetime
from tqdm import tqdm
from torch import optim
import torch.backends.cudnn as cudnn
import pickle
cudnn.benchmark = True
cudnn.deterministic = True
seed_everything(0) 

gpu_ids = [0]


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda:0"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda:0"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())

bias_layers = [BiasLayer() for _ in range(loader.episode_length)]
bias_layers = [nn.DataParallel(bias, device_ids=gpu_ids).cuda() for bias in bias_layers] 

def bias_forward(input, bias_layers, episode_length, num_class):
    inputs = [input[:, i*(num_class // episode_length):(i+1)*(num_class // episode_length)] for i in range(num_class)]
    outs = [bias_layers[i](inputs[i]) for i in range(episode_length)]
    return torch.cat(outs, dim = 1)



def update_memory(memory, dl, memory_size=1800):
    memory_t = set(memory.dataset.labels)
    dl_t = set(dl.dataset.labels)
    dl.dataset.labels = list(dl.dataset.labels)
    if len(dl_t.difference(memory_t)) == 0:
        return memory
    else:
        unique_data = list(set([(p,t) for p,t in zip(memory.dataset.datas, memory.dataset.labels)] + [(p,t) for p,t in zip(dl.dataset.datas, dl.dataset.labels)]))
        #  print(memory.dataset.labels)
        categorized_unique_data = {i: [] for i in list(set(memory.dataset.labels + dl.dataset.labels))}
        for p, t in unique_data:
            categorized_unique_data[t].append(p)
        num_per_class = int(memory_size / len(set(memory.dataset.labels + dl.dataset.labels)))
        datas, labels = [], []
        for t in list(set(memory.dataset.labels + dl.dataset.labels)):
            tmp = categorized_unique_data[t]
            random.shuffle(tmp)
            tmp = tmp[:num_per_class]
            datas.extend(tmp)
            labels.extend([t] * len(tmp))
        memory.dataset.datas, memory.dataset.labels = datas, labels


totalNet = TotalNet()
feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids).cuda()
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids).cuda()
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids).cuda()
discriminator_separate = nn.DataParallel(totalNet.discriminator_separate, device_ids=gpu_ids).cuda()



# ===================optimizer
#  optimizer_finetune = optim.SGD(feature_extractor.parameters(), lr=args.train.lr10)
optimizer_finetune = optim.SGD(feature_extractor.parameters(), lr=args.train.lr/10, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
#  optimizer_cls = optim.SGD(classifier.parameters(), lr=args.train.lr)
optimizer_cls = optim.SGD(classifier.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
#  optimizer_discriminator = optim.SGD(discriminator.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
optimizer_discriminator_separate = optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)
#  optimizer_discriminator_separate = optim.SGD(discriminator_separate.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay, nesterov=True)


best_acc = 0


E = []
for e in range(loader.episode_length):
    for _ in range(args.misc.episode_per_epoch):
        #  E.append(3)
        E.append(e)

for epoch_id, e in enumerate(E):
    print('##########################################################')
    print('episode {}'.format(e))
    print('##########################################################')
    source_train_dl, source_val_dl, source_test_dl, target_train_dl, target_test_dl = loader(e)
   
    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id-e*args.misc.episode_per_epoch} ', total=min(len(source_train_dl), len(target_train_dl)))
    bias_optimizer = optim.Adam(bias_layers[e].parameters(), lr=0.001)

    adv_losses = []
    adv_separate_losses = []
    ce_losses = []
    for it, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):
        save_label_target = label_target  # for debug usage

        label_source = label_source.cuda()
        label_target = label_target.cuda()
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.cuda()

        im_target = im_target.cuda()
        fc1_s = feature_extractor.forward(im_source)
        #  feature_extractor.eval()
        fc1_t = feature_extractor.forward(im_target)
        #  feature_extractor.train()
        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)
        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        domain_prob_discriminator_source_separate = discriminator_separate.forward(feature_source.detach())
        domain_prob_discriminator_target_separate = discriminator_separate.forward(feature_target.detach())

        source_share_weight = get_source_share_weight(domain_prob_discriminator_source_separate, fc2_s, domain_temperature=1.0, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = get_target_share_weight(domain_prob_discriminator_target_separate, fc2_t, domain_temperature=1.0, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)

        #  ==============================compute loss
        adv_loss = torch.zeros(1, 1).cuda()
        adv_loss_separate = torch.zeros(1, 1).cuda()
        if e == 0 :
            tmp = source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        else:
            tmp = 0.5 * source_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)
        tmp = target_share_weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))
        adv_loss += torch.mean(tmp, dim=0, keepdim=True)

        if e == 0 :
            adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
        else:
            adv_loss_separate += 0.5 * nn.BCELoss()(domain_prob_discriminator_source_separate, torch.ones_like(domain_prob_discriminator_source_separate))
        adv_loss_separate += nn.BCELoss()(domain_prob_discriminator_target_separate, torch.zeros_like(domain_prob_discriminator_target_separate))
#
        #  ============================== cross entropy loss, it receives logits as its inputs
        #  ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source)
        #  ce = torch.mean(ce, dim=0, keepdim=True)
        ce = nn.CrossEntropyLoss()(bias_forward(fc2_s, bias_layers, loader.episode_length, args.data.dataset.n_share), label_source)
        
        if e != 0:
            # ============================== previsou episodes
            for pre_im_source, pre_label_source in memory_dl:
                break
            pre_im_source = pre_im_source.cuda()
            pre_label_source = pre_label_source.long().cuda()
            fc1_ps = feature_extractor.forward(pre_im_source)

            fc1_ps, feature_source_p, fc2_ps, predict_prob_source_p = classifier.forward(fc1_ps)
            ce_p = nn.CrossEntropyLoss()(bias_forward(fc2_ps, bias_layers, loader.episode_length, args.data.dataset.n_share), pre_label_source)

            domain_prob_discriminator_source_p = discriminator.forward(feature_source_p)

            domain_prob_discriminator_source_separate_p = discriminator_separate.forward(feature_source_p.detach())

            source_share_weight_p = get_source_share_weight(domain_prob_discriminator_source_separate_p, fc2_ps, domain_temperature=1.0, class_temperature=10.0)
            source_share_weight_p = normalize_weight(source_share_weight_p)
            adv_loss_separate += 0.5*nn.BCELoss()(domain_prob_discriminator_source_separate_p, torch.ones_like(domain_prob_discriminator_source_separate_p))
 
            tmp = source_share_weight_p * nn.BCELoss(reduction='none')(domain_prob_discriminator_source_p, torch.ones_like(domain_prob_discriminator_source_p))
            adv_loss += 0.5*torch.mean(tmp, dim=0, keepdim=True)

            ce = 0.5*(ce+ce_p)

        adv_losses.append(adv_loss.item())
        ce_losses.append(ce.item())
        adv_separate_losses.append(adv_loss_separate.item())

        loss = ce + adv_loss + adv_loss_separate
        optimizer_finetune.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_discriminator.zero_grad()
        optimizer_discriminator_separate.zero_grad()
        loss.backward()
        optimizer_finetune.step()
        optimizer_cls.step()
        optimizer_discriminator.step()
        optimizer_discriminator_separate.step()

    print("adv loss : ", np.mean(adv_losses))
    print("ce loss : ", np.mean(ce_losses))
    print("adv separate loss : ", np.mean(adv_separate_losses))



    if epoch_id+1 == len(E) or E[epoch_id+1] != e :
        # learn only bias_layers
        update_memory(memory_dl, source_train_dl, memory_size=1800)
        update_memory(memory_val_dl, source_val_dl, memory_size=200)
        if e > 0:
            # for _ in range(20):
            #     for im_source, label_source in memory_val_dl:
            #         break
            for _ in range(4):
                for im_source, label_source in memory_val_dl:
                    bias_layers[e].zero_grad()
                    im_source = im_source.cuda()
                    label_source = label_source.long().cuda()
                    feature_extractor.eval()
                    fc1_s = feature_extractor.forward(im_source)
                    feature_extractor.train()
                    _, _, fc2_s, _ = classifier.forward(fc1_s)
                    ce = nn.CrossEntropyLoss(reduction='none')(bias_forward(fc2_s, bias_layers, loader.episode_length, args.data.dataset.n_share), label_source)
                    ce = torch.mean(ce, dim=0, keepdim=True)
                    ce.backward()
                    bias_optimizer.step()

        #  if e == 3:
        feature_extractor.eval()
        classifier.eval()
        discriminator.eval()
        discriminator_separate.eval()
        eval_source(bias_layers, feature_extractor, classifier, discriminator, discriminator_separate, source_test_dl, e)
        domain_probs, labels = eval_cosda(bias_layers, feature_extractor, classifier, discriminator, discriminator_separate, target_test_dl, e)
        eval_domain(domain_probs, labels, target_test_dl, e)
        feature_extractor.train()
        classifier.train()
        discriminator.train()
        discriminator_separate.train()
        
        # save model

        # torch.save(totalNet.state_dict(),'bic.model.pth') 
        #  torch.save(discriminator_separate.state_dict(),'dis.model.pth')
        # for i in range(len(bias_layers)):
            # torch.save(bias_layers[i].state_dict(), f'bias_layer{i}.pth')

    #  for i in range(len(bias_layers)):
        #  bias_layers[i].printParam(i)
        # update_memory(memory_dl, source_train_dl, memory_size=1800)
        # update_memory(memory_val_dl, source_val_dl, memory_size=200)
        #  assert len(memory_dl.dataset.datas) <= 1800




