import torch
import torch.nn as nn


class DeepFM(nn.Module):
    
    def __init__(self, num_of_users, num_of_times, num_of_locs, rank=8, latent_dim=64):
        
        super(DeepFM, self).__init__()
        
        self.num_of_users = num_of_users
        self.num_of_times = num_of_times
        self.num_of_locs = num_of_locs
        self.rank = rank
        self.latent_dim = latent_dim
        
        self.user_embedding = nn.Embedding(num_of_users, rank * latent_dim)
        self.time_embedding = nn.Embedding(num_of_times, rank * latent_dim)
        
        self.user_linear_weights = nn.Embedding(num_of_users, latent_dim)
        self.time_linear_weights = nn.Embedding(num_of_times, latent_dim)
        
        self.output_softmax = nn.Sequential(
            nn.Linear(latent_dim, (latent_dim + num_of_locs) // 2),
            nn.LeakyReLU(),
            nn.Linear((latent_dim + num_of_locs) // 2, num_of_locs),
        )
        
        self.criteria = nn.CrossEntropyLoss(reduction='sum')
        
    def predict(self, x_u, x_t):
        embed_u = self.user_embedding(x_u).view(-1, 1, self.rank)
        embed_t = self.time_embedding(x_t).view(-1, self.rank, 1)
        interact_term = torch.bmm(embed_u, embed_t).view(-1, self.latent_dim)
        linear_term = self.user_linear_weights(x_u) + self.time_linear_weights(x_t)
        return self.output_softmax(interact_term + linear_term)
    
    def predict_prob(self, x_u, x_t):
        return nn.functional.softmax(self.predict(x_u, x_t), dim=-1)
        
    def forward(self, x_u, x_t, y_l):
        return self.criteria(self.predict(x_u, x_t), y_l)
    
    
class DeepFMDayTime(nn.Module):
    
    def __init__(self, num_of_users, num_of_tofds, num_of_days, num_of_locs, rank=8, latent_dim=64):
        
        super(DeepFMDayTime, self).__init__()
        
        self.num_of_users = num_of_users
        self.num_of_locs = num_of_locs
        self.num_of_tofds = num_of_tofds
        self.num_of_days = num_of_days
        self.rank = rank
        self.latent_dim = latent_dim
        
        self.user_embedding = nn.Embedding(num_of_users, rank * latent_dim)
        self.tofd_embedding = nn.Embedding(num_of_tofds, rank * latent_dim)
        self.day_embedding = nn.Embedding(num_of_days, rank * latent_dim)
        
        self.user_linear_weights = nn.Embedding(num_of_users, latent_dim)
        self.tofd_linear_weights = nn.Embedding(num_of_tofds, latent_dim)
        self.day_linear_weights = nn.Embedding(num_of_days, latent_dim)
        
        self.output_softmax = nn.Sequential(
            nn.Linear(latent_dim, (latent_dim + num_of_locs) // 2),
            nn.LeakyReLU(),
            nn.Linear((latent_dim + num_of_locs) // 2, num_of_locs),
        )
        
        self.criteria = nn.CrossEntropyLoss(reduction='sum')
        
    def predict(self, x_u, x_d, x_t):
        embed_u = self.user_embedding(x_u).view(-1, self.rank)
        embed_d = self.day_embedding(x_d).view(-1, self.rank)
        embed_t = self.tofd_embedding(x_t).view(-1, self.rank)
        interact_term = torch.bmm(embed_u.view(-1, 1, self.rank), embed_d.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_u.view(-1, 1, self.rank), embed_t.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_d.view(-1, 1, self.rank), embed_t.view(-1, self.rank, 1)).view(-1, self.latent_dim)
        linear_term = self.user_linear_weights(x_u) + self.day_linear_weights(x_d) + self.tofd_linear_weights(x_t)
        return self.output_softmax(interact_term + linear_term)
    
    def predict_prob(self, x_u, x_d, x_t):
        return nn.functional.softmax(self.predict(x_u, x_d, x_t), dim=-1)
        
    def forward(self, x_u, x_d, x_t, y_l):
        return self.criteria(self.predict(x_u, x_d, x_t), y_l)
    
    
class DeepFM3Times(nn.Module):
    
    def __init__(self, num_of_users, num_of_times, num_of_tofds, num_of_days, num_of_locs, rank=8, latent_dim=64):
        
        super(DeepFM3Times, self).__init__()
        
        self.num_of_users = num_of_users
        self.num_of_times = num_of_times
        self.num_of_locs = num_of_locs
        self.num_of_tofds = num_of_tofds
        self.num_of_days = num_of_days
        self.rank = rank
        self.latent_dim = latent_dim
        
        self.user_embedding = nn.Embedding(num_of_users, rank * latent_dim)
        self.time_embedding = nn.Embedding(num_of_times, rank * latent_dim)
        self.tofd_embedding = nn.Embedding(num_of_tofds, rank * latent_dim)
        self.day_embedding = nn.Embedding(num_of_days, rank * latent_dim)
        
        self.user_linear_weights = nn.Embedding(num_of_users, latent_dim)
        self.time_linear_weights = nn.Embedding(num_of_times, latent_dim)
        self.tofd_linear_weights = nn.Embedding(num_of_tofds, latent_dim)
        self.day_linear_weights = nn.Embedding(num_of_days, latent_dim)
        
        self.output_softmax = nn.Sequential(
            nn.Linear(latent_dim, (latent_dim + num_of_locs) // 2),
            nn.LeakyReLU(),
            nn.Linear((latent_dim + num_of_locs) // 2, num_of_locs),
        )
        
        self.criteria = nn.CrossEntropyLoss(reduction='sum')
        
    def predict(self, x_u, x_dt, x_d, x_t):
        embed_u = self.user_embedding(x_u).view(-1, self.rank)
        embed_dt = self.time_embedding(x_dt).view(-1, self.rank)
        embed_d = self.day_embedding(x_d).view(-1, self.rank)
        embed_t = self.tofd_embedding(x_t).view(-1, self.rank)
        interact_term = torch.bmm(embed_u.view(-1, 1, self.rank), embed_dt.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_u.view(-1, 1, self.rank), embed_d.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_u.view(-1, 1, self.rank), embed_t.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_dt.view(-1, 1, self.rank), embed_d.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_dt.view(-1, 1, self.rank), embed_t.view(-1, self.rank, 1)).view(-1, self.latent_dim) + \
                        torch.bmm(embed_d.view(-1, 1, self.rank), embed_t.view(-1, self.rank, 1)).view(-1, self.latent_dim)
        linear_term = self.user_linear_weights(x_u) + self.time_linear_weights(x_dt) + self.tofd_linear_weights(x_t) + self.day_linear_weights(x_d)
        return self.output_softmax(interact_term + linear_term)
    
    def predict_prob(self, x_u, x_dt, x_d, x_t):
        return nn.functional.softmax(self.predict(x_u, x_dt, x_d, x_t), dim=-1)
        
    def forward(self, x_u, x_dt, x_d, x_t, y_l):
        return self.criteria(self.predict(x_u, x_dt, x_d, x_t), y_l)