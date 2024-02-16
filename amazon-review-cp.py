import numpy as np
import os
from tqdm import tqdm
import pickle
import dgl 
import torch
import time
from urllib import request
import urllib
from transformers import BertModel, BertForPreTraining, BertConfig
import pdb


meta_file_path = "/home/ubuntu/amazon-review/All_Amazon_Meta.json.gz"
review_file_path = "/home/ubuntu/amazon-review/All_Amazon_Review_5.json.gz"
save_node_meta_data_path = "node_meta_data.pkl"
save_edge_path = "edges.pkl"
save_node_path = "nodes.pkl"
output_path = "/home/ubuntu/feature-discrepancy/data/amazon-review-images"
os.makedirs(output_path, exist_ok=True)

def save_to_pickle(save_list, save_path):
    pickle_out = open(save_path, "wb")
    pickle.dump(save_list, pickle_out)
    pickle_out.close()

def load_pickle(save_path):
    pickle_in = open(save_path, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    return data

edges = load_pickle(save_edge_path)
nodes = load_pickle(save_node_path)
node_meta_data = load_pickle(save_node_meta_data_path)

node_ids = {}
counter = 0
node_id_edges = []
# renumber edges according to node ids
for e in edges:
    u = e[0]
    v = e[1]
    if u not in node_ids:
        node_ids[u] = counter
        counter = counter + 1
    if v not in node_ids:
        node_ids[v] = counter
        counter = counter + 1
    node_id_edges.append([node_ids[u], node_ids[v]])

node_id_edges = torch.tensor(np.array(node_id_edges))
g = dgl.graph((node_id_edges[:,0], node_id_edges[:,1]))

stats = {
    'not_found': 0
}


#### extract text feature
text_feat_list = []
image_feat_list = []
inv_node_ids = {v: k for k, v in node_ids.items()}
for i in tqdm(range(g.number_of_nodes())):
    asin = inv_node_ids[i]
    obj = node_meta_data[asin]
    text_f = obj['description']
    text_feat_list.append(text_f[0])
    image_f = obj['image']
    number_of_features = len(image_f)
    number_of_features = min(number_of_features, 1)  # only save first one
    for jj in range(number_of_features):
        image_url = image_f[jj]
        suffix = image_url.split('.')[-1]
        image_url_hr = '.'.join(image_url.split('.')[:-2]) + '.' + suffix
        save_path = os.path.join(output_path, '{}_{:0>2}.{}'.format(asin, jj, suffix))
        if os.path.exists(save_path):
            print("skip {}".format(image_url_hr))
            continue
        try:
            request.urlretrieve(image_url_hr, save_path)
        #except urllib.error.HTTPError:
        except:
            print("can't find {}".format(image_url_hr))
            stats['not_found'] += 1
            continue

        time.sleep(0.1)
    tqdm.write("node" + str(i)+ "completed.")
    image_location = os.path.join(output_path, '{}'.format(asin))
    #image_feat_list.append(image_feat_list)

save_text_path = "/home/ubuntu/feature-discrepancy/data/amazon-review-text-feats.pkl"
save_image_path = "/home/ubuntu/feature-discrepancy/data/amazon-review-image-feats.pkl"
save_to_pickle(text_feat_list, save_text_path)
save_to_pickle(image_feat_list, save_image_path)
print("Construct successful!")

pdb.set_trace()

# request.urlretrieve(image_url, os.path.join(output_dir, '{}.{}'.format(keyword, suffix)))
# time.sleep(0.1)