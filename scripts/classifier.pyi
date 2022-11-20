
tokenized_train = tokenizer(train_texts, padding = True, truncation = True, return_tensors="pt")
tokenized_val = tokenizer(dev_texts , padding = True, truncation = True,  return_tensors="pt")

print(tokenized_train.keys())

#move on device (GPU)
tokenized_train = {k:torch.tensor(v).to(device) for k,v in tokenized_train.items()}
tokenized_val = {k:torch.tensor(v).to(device) for k,v in tokenized_val.items()}

with torch.no_grad():
  hidden_train = clean_model(**tokenized_train) #dim : [batch_size(nr_sentences), tokens, emb_dim]
  hidden_val = clean_model(**tokenized_val)

#get only the [CLS] hidden states
cls_train = hidden_train.last_hidden_state[:,0,:]
cls_val = hidden_val.last_hidden_state[:,0,:]
x_train = cls_train.to("cpu")
x_val = cls_val.to("cpu")


print(x_train.shape, train_labels.shape, x_val.shape, dev_labels.shape)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train,train_labels)
rf.score(x_val,dev_labels)

