from transformers import AutoTokenizer, AutoFeatureExtractor, ClapModel
import torch
import torch.nn as nn
import torch.nn.functional as F

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

inputs = tokenizer(["the sound of a dog"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)

random_audio = torch.rand((16_000))
inputs = feature_extractor(random_audio, return_tensors="pt")
audio_features = model.get_audio_features(**inputs)

class MM(nn.Module):
    def __init__(self):
        super(MM, self).__init__()  
        
        self.drop20 = nn.Dropout(p=0.2)
        self.drop5 = nn.Dropout(p=0.05) 
        
        self.dense_drob_512 = nn.Linear(768, 512)
        
        self.gen_key_L1 = nn.Linear(512, 256) # 512X256
        self.gen_query_L1 = nn.Linear(512, 256) # 512X256
        self.gen_key_L2 = nn.Linear(512, 256) # 512X256
        self.gen_query_L2 = nn.Linear(512, 256) # 512X256

        self.soft = nn.Softmax(dim=1)

        self.dense1 = nn.Linear(1024, 512) # 512X256
        self.dense2 = nn.Linear(1024, 512) # 512X256
        
        self.fc_out = nn.Linear(512, 256) # 512X256
        
    def selfattNFuse_L1(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L1(vec1))
            k1 = F.relu(self.gen_key_L1(vec1))
            q2 = F.relu(self.gen_query_L1(vec2))
            k2 = F.relu(self.gen_key_L1(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.dense1(torch.cat((wtd_i1,wtd_i2), 1)))

            return out_rep
    
    def selfattNFuse_L2(self, vec1, vec2): 
            q1 = F.relu(self.gen_query_L2(vec1))
            k1 = F.relu(self.gen_key_L2(vec1))
            q2 = F.relu(self.gen_query_L2(vec2))
            k2 = F.relu(self.gen_key_L2(vec2))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = vec1 * prob_1[:, None]
            wtd_i2 = vec2 * prob_2[:, None]
            out_rep = F.relu(self.dense2(torch.cat((wtd_i1,wtd_i2), 1)))
            return out_rep
    
    def forward(self, audio_rep, text_rep):        

        fused_audio_text = self.selfattNFuse_L1(audio_rep, text_rep)

        final_out = F.relu(self.fc_out(fused_audio_text))
        
        return final_out
    
model = MM()
out = model(audio_features, text_features)
print(out.shape)