from easydl import *
from tqdm import tqdm
from config import *
from lib import *
from data import *
import torch.backends.cudnn as cudnn
import pickle
cudnn.benchmark = True
cudnn.deterministic = True

def outlier(each_target_share_weight, w_0):
    return each_target_share_weight < w_0

def bias_forward(input, bias_layers, episode_length, num_class):
    inputs = [input[:, i*(num_class // episode_length):(i+1)*(num_class // episode_length)] for i in range(num_class)]
    outs = [bias_layers[i](inputs[i]) for i in range(episode_length)]
    return torch.cat(outs, dim = 1)



def get_acc(counter_topk, counter_num, num):
    cats = list(range(num*16+16)) 
    accs = []
    for cat in cats:
       accs.append(counter_topk[cat]/counter_num[cat]) 
    #  print(accs)
    return np.mean(accs)



if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = [0]
    output_device = gpu_ids[0]

def eval_source(bias_layers, feature_extractor, classifier, discriminator, discriminator_separate, source_test_dl, e):
    source_pred_probs = []
    source_labels = []
    with torch.no_grad():
        for i, (im, label) in enumerate(tqdm(source_test_dl, desc='testing ')):
            im = im.to(output_device)
            source_label = label.to(output_device)

            feature = feature_extractor.forward(im)
            feature, _, before_softmax, source_pred_prob = classifier.forward(feature)
            s_output = bias_forward(before_softmax, bias_layers, 10, 160)
            source_pred_prob = s_output.cpu()
            for b in range(source_label.size(0)):
                source_pred_probs.append(source_pred_prob[b])
                source_labels.append(source_label[b].cpu().numpy())

    counter_top1 =[0]*(args.data.dataset.n_total-1)
    counter_num = [0]*args.data.dataset.n_total
    for (each_pred_prob, each_label) in zip(source_pred_probs, source_labels):
        each_label = each_label.item()
        counter_num[each_label] += 1
        pred_id_top1 = np.argmax(each_pred_prob)
        if each_label == pred_id_top1:
            counter_top1[each_label] += 1
    print("source top1 acc : ", 100*get_acc(counter_top1, counter_num, e))


def eval_domain(domain_pred_probs, domain_labels, domain_test_dl, e):

    counter_top1 =[0]*(args.data.dataset.n_total-1)
    counter_top5 =[0]*(args.data.dataset.n_total-1)
    counter_top10 =[0]*(args.data.dataset.n_total-1)

    counter_num = [0]*args.data.dataset.n_total
    for (each_pred_prob, each_label) in zip(domain_pred_probs, domain_labels):
        each_label = each_label.item()
        counter_num[each_label] += 1
        pred_id_top1 = np.argmax(each_pred_prob)
        if each_label == pred_id_top1:
            counter_top1[each_label] += 1
        pred_id_top5 = np.argsort(each_pred_prob)[-5:][::-1]
        if each_label in pred_id_top5:
            counter_top5[each_label] += 1
        pred_id_top10 = np.argsort(each_pred_prob)[-10:][::-1]
        if each_label in pred_id_top10:
            counter_top10[each_label] += 1
    print("domain top1 acc : ", 100*get_acc(counter_top1, counter_num, e))
    print("domain top5 acc : ", 100*get_acc(counter_top5, counter_num, e))
    print("domain top10 acc : ", 100*get_acc(counter_top10, counter_num, e))



def eval_cosda(bias_layers, feature_extractor, classifier, discriminator, discriminator_separate, target_test_dl, e): 
    labels = []
    predict_probs = []
    target_share_weights = []
    with torch.no_grad():

        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor.forward(im)
            feature, __, before_softmax, predict_prob = classifier.forward(feature)
            s_output = bias_forward(before_softmax, bias_layers, 10, 160)
            domain_prob = discriminator_separate.forward(__)
            
            target_share_weight = get_target_share_weight(domain_prob, before_softmax, domain_temperature=1.0, class_temperature=1.0)
            predict_prob = s_output.cpu().numpy()
            label = label.cpu().numpy()
            target_share_weight = target_share_weight.cpu().numpy()

            for b in range(len(label)):
                predict_probs.append(predict_prob[b])
                labels.append(label[b])
                target_share_weights.append(target_share_weight[b])


    w_0_left = -1
    w_0_right = 1
    outlier_acc = 0
    while outlier_acc < 90 or outlier_acc > 90.1:
        w_0_mid = (w_0_left + w_0_right)/2 
        counter_mat = [[0]*args.data.dataset.n_total for i in range(args.data.dataset.n_total)]
        counter_num = [0]*args.data.dataset.n_total
        counter_top1 =[0]*(args.data.dataset.n_total-1)
        counter_top5 =[0]*(args.data.dataset.n_total-1)
        counter_top10 =[0]*(args.data.dataset.n_total-1)
        counter_outlier = 0
        for (each_pred_prob, each_label, each_target_share_weight) in zip(predict_probs, labels, target_share_weights):
            each_label = each_label.item()
            counter_num[each_label] += 1
            if not outlier(each_target_share_weight[0], w_0_mid):
            #  if True:
                pred_id_top1 = np.argmax(each_pred_prob)
                if each_label == pred_id_top1:
                    counter_top1[each_label] += 1
                pred_id_top5 = np.argsort(each_pred_prob)[-5:][::-1]
                if each_label in pred_id_top5:
                    counter_top5[each_label] += 1
                pred_id_top10 = np.argsort(each_pred_prob)[-10:][::-1]
                if each_label in pred_id_top10:
                    counter_top10[each_label] += 1
                counter_mat[each_label][pred_id_top1] += 1
                assert pred_id_top1 < 160
            else:
                counter_mat[each_label][160] += 1
                if each_label == 160:
                    counter_outlier += 1
        outlier_acc = 100*counter_outlier/counter_num[-1]
        if outlier_acc < 90:
            w_0_left = w_0_mid
        else:
            w_0_right = w_0_mid
    #  print(np.matrix(counter_mat))
    print("w_0 : ", w_0_mid)
    print("top1 acc : ", 100*get_acc(counter_top1, counter_num, e))
    print("top5 acc : ", 100*get_acc(counter_top5, counter_num, e)) 
    print("top10 acc : ", 100*get_acc(counter_top10, counter_num, e)) 
    print("outlier acc : ", 100*counter_outlier/counter_num[-1])
    return predict_probs, labels
