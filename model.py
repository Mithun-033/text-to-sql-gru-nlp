import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import random
from torchinfo import summary

df=pd.read_csv("hf://datasets/Mudasir692/text-to-sql/text_to_sql_dataset.csv")
print("dataset loaded")

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X=df["sentence"]
X=[re.sub(r"[^a-zA-Z0-9\s]","",i.lower().strip()) for i in X]
y=df["sql"]
y=[i.lower() for i in y]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

idx=4
word2id={"<PAD>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3}
for i in X:
    for j in i.split():
        if j not in word2id:
            word2id[j]=idx
            idx+=1

for i in y:
    temp=i.split()
    temp.insert(0,"<BOS>")
    temp.append("<EOS>")
    for j in temp:
        if j not in word2id:
            word2id[j]=idx
            idx+=1
id2word_sql={v:k for k,v in word2id.items()}

def stoi_text(sentence):
    return [word2id.get(i, word2id["<UNK>"]) for i in sentence.split()]

def stoi_sql(sentence):
    temp=sentence.split()
    temp.insert(0,"<BOS>")
    temp.append("<EOS>")
    return [word2id[i] for i in temp]

x_train=[stoi_text(i) for i in x_train]
x_test=[stoi_text(i) for i in x_test]
y_train=[stoi_sql(i) for i in y_train]
y_test=[stoi_sql(i) for i in y_test]

def generator(X,y,batch_size=20):
    for i in range(0,len(X),batch_size):
        X_batch=X[i:i+batch_size]
        y_batch=y[i:i+batch_size]

        max_len_x=max(len(i) for i in X_batch)
        max_len_y=max(len(i) for i in y_batch)

        X_pad=[i+[0]*(max_len_x-len(i)) for i in X_batch]
        y_pad=[i+[0]*(max_len_y-len(i)) for i in y_batch]

        yield torch.tensor(X_pad),torch.tensor(y_pad)

class GRU(nn.Module):
    def __init__(self,n_text,n_sql,n_text_embed,n_sql_embed,n_hidden,n_layer):
        super(GRU,self).__init__()

        self.N_text=n_text
        self.N_sql=n_sql
        self.D_text=n_text_embed
        self.D_sql=n_sql_embed
        self.M=n_hidden
        self.L=n_layer

        self.text_embed=nn.Embedding(self.N_text,self.D_text,padding_idx=word2id["<PAD>"])
        self.text_dropout=nn.Dropout(0.2)

        self.text_gru=nn.GRU(
            input_size=self.D_text,
            hidden_size=self.M,
            batch_first=True,
            num_layers=self.L,
            dropout=0.2
        )

        self.sql_embed=nn.Embedding(self.N_sql,self.D_sql,padding_idx=word2id["<PAD>"])
        self.sql_dropout=nn.Dropout(0.2)

        self.sql_gru=nn.GRU(
            input_size=self.D_sql,
            hidden_size=self.M,
            batch_first=True,
            num_layers=self.L,
            dropout=0.2
        )

        self.out_dropout=nn.Dropout(0.1)
        self.dense=nn.Linear(self.M,self.N_sql)

    def forward(self,x,y):
        h0=torch.zeros(self.L,x.size(0),self.M).to(device)

        text_vectors=self.text_dropout(self.text_embed(x))
        _,last_hidden_state=self.text_gru(text_vectors,h0)

        sql_vectors=self.sql_dropout(self.sql_embed(y))
        out,_=self.sql_gru(sql_vectors,last_hidden_state)

        out=self.out_dropout(out)
        output=self.dense(out)
        return output
    
    def summary(self,input_size):
        return summary(self,input_size=input_size)
    
    def generate(self,input_sentence):
        self.eval()
        x=re.sub(r"[^a-zA-Z0-9\s]","",input_sentence.lower().strip())
        x=stoi_text(x)
        x=torch.tensor(x).unsqueeze(0).to(device)
        h0=torch.zeros(self.L,1,self.M).to(device)

        text_vec=self.text_embed(x)
        _,hidden=self.text_gru(text_vec,h0)

        curr_token=torch.tensor([[word2id["<BOS>"]]],device=device)
        output_sentence=[]

        for _ in range(50):
            y_vec=self.sql_embed(curr_token)
            out,hidden=self.sql_gru(y_vec,hidden)
            logits=self.dense(out[:,-1,:])
            input_ids = set(stoi_text(input_sentence))
            bias=1.5
            for tok in input_ids:
                logits[0, tok] += bias
            next_token=logits.argmax(-1).item()

            if next_token==word2id["<EOS>"]:
                break
            output_sentence.append(next_token)
            curr_token=torch.tensor([[next_token]],device=device)
        return ' '.join([id2word_sql[i] for i in output_sentence])
        

epochs=20
batch_size=16
model=GRU(len(word2id),len(word2id),64,64,140,3).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs*20
)
criterion=nn.CrossEntropyLoss(ignore_index=word2id["<PAD>"])
train_loss=[]
test_loss=[]
for epoch in range(epochs):
    model.train()
    step=0
    sample_train=0
    random.seed(epoch)
    idx=list(range(len(x_train)))
    random.shuffle(idx)
    x_train[:]=[x_train[i] for i in idx]
    y_train[:]=[y_train[i] for i in idx]
    for x_batch,y_batch in generator(x_train,y_train,batch_size):
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)

        optimizer.zero_grad()
        prediction=model(x_batch,y_batch[:,:-1])
        loss=criterion(prediction.reshape(-1,prediction.size(-1)),y_batch[:,1:].reshape(-1,))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        sample_train+=batch_size
        step+=1
        print(f'Epoch:{epoch+1} Samples:{sample_train}, Loss:{loss.item()}')
    train_loss.append(loss.item())
    
    model.eval()
    total_loss=0
    count=0
    sample_test=0
    with torch.no_grad():
        for x_batch,y_batch in generator(x_test,y_test,batch_size):
            x_batch=x_batch.to(device)
            y_batch=y_batch.to(device)

            prediction=model(x_batch,y_batch[:,:-1])
            loss=criterion(prediction.reshape(-1,prediction.size(-1)),y_batch[:,1:].reshape(-1,))
            total_loss+=loss.item()
            count+=1
            sample_test+=batch_size
            if count%10==0:
                print(f'Epoch:{epoch+1} Samples:{sample_test}, Loss:{loss.item()}')
        test_loss.append((total_loss/count))
        
plt.plot(range(epochs),train_loss,label="Train Loss")
plt.plot(range(epochs),test_loss,label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve_gen.png")
plt.close()

torch.save(model.state_dict(),"text_to_sql_model_gen.pth")

while True:
    x=input("Enter text :")
    print(model.generate(x))    


