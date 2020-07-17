
load('WS.mat');


Para.M = 500;
Para.K = 100;
Para.is_meta_data = 1;
Para.is_sample_alpha = 0;
Para.max_iter = 2000;
Para.word_train_prop = 101;

model = MIGA(doc, doc_label, Para);

save_dir = './save';
if ~exist(save_dir,'dir')
    mkdir(save_dir);
end
save(sprintf('%s/model.mat',save_dir),'model');

show_cluster_label_doc_topic;