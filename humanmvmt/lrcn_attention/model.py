import torch
import torch.nn as nn
from torch.nn import functional as F


class LSTM_B1(nn.Module):
    def __init__(self, num_channels, segment_size, classes):
        super(LSTM_B1, self).__init__()
        self.lstm_bi_do = torch.nn.LSTM(input_size=4, hidden_size=100, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(200, classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 
        self.num_classes = classes

    def forward(self, sequence):
        #feature_maps = []
        #for segment in sequence:
        #    feature_maps.append(self.cnn_block.forward(segment))
        #feature_maps = torch.stack(feature_maps)
        #print(sequence.shape)
        feature_maps = sequence.permute(0,2,1)

        #feature_maps = torch.reshape(sequence,(a1,b1,-1))
        out, (_, _) = self.lstm_bi_do(feature_maps)
        #print(out.shape)

        out = self.linear(out[:,-1,:])
        #out = torch.reshape(out, (-1,1,self.num_classes))
        output = self.dropout(out)
        output = self.logSoftmax(output)
        #print(output.shape)
        #print(output)
        #exit()
        #output = output.permute(0, 2, 1)
        return out, output

class LSTM_B(nn.Module):
    def __init__(self, num_channels, segment_size, classes):
        super(LSTM_B, self).__init__()
        self.lstm_bi_do = torch.nn.LSTM(input_size=4, hidden_size=50, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(100, classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 
        self.classes = classes

    def forward(self, sequence):
        #feature_maps = []
        #for segment in sequence:
        #    feature_maps.append(self.cnn_block.forward(segment))
        #feature_maps = torch.stack(feature_maps)
        #print(sequence.shape)
        feature_maps = sequence.permute(0,2,1)

        #feature_maps = torch.reshape(sequence,(a1,b1,-1))
        out, (_, _) = self.lstm_bi_do(feature_maps)
        #print(out.shape)

        out = self.linear(out[:,-1,:])
        #out = torch.reshape(out, (a1,b1,self.classes))
        output = self.dropout(out)
        output = self.logSoftmax(output)
        #output = output.permute(0, 2, 1)
        return out, output

class LSTM(nn.Module):
    def __init__(self, num_channels, segment_size, classes):
        super(LSTM, self).__init__()
        self.lstm_bi_do = torch.nn.LSTM(input_size=160, hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, classes)
        self.logSoftmax = nn.LogSoftmax(dim=2) 

    def forward(self, sequence):
        #feature_maps = []
        #for segment in sequence:
        #    feature_maps.append(self.cnn_block.forward(segment))
        #feature_maps = torch.stack(feature_maps)
        [a1,b1,c1,d1] = sequence.shape
        sequence = torch.reshape(sequence, (a1,b1, -1))
        out, (_, _) = self.lstm_bi_do(sequence)
        #print(f"feature.shape:{feature_maps.shape}")
        out = self.linear1(out)
        out = self.linear2(out)
        output = self.logSoftmax(out)
        output = output.permute(0, 2, 1)
        return out, output
        #return out, output.view(output.shape[0], output.shape[2], output.shape[1])

class LRCN_A(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_A, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_si = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=1, bidirectional=False, batch_first=True)
        self.linear = torch.nn.Linear(512, num_classes)

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_si(feature_maps)
        output = self.linear(out)
        return output.view(output.shape[0], output.shape[2], output.shape[1])


class LRCN_B(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_B, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_si = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=3, bidirectional=False, batch_first=True)
        self.linear = torch.nn.Linear(512, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_si(feature_maps)
        out = self.linear(out)
        output = self.logSoftmax(out)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_C(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_C, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_si = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=3, bidirectional=False, batch_first=True)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_si(feature_maps)
        out = self.linear1(out)
        output = self.linear2(out)
        output = self.linear3(output)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_D(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_D, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_si_dropout = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=3, 
                                             bidirectional=False, batch_first=True, dropout = 0.3)
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_si_dropout(feature_maps)
        out = self.linear1(out)
        output = self.linear2(out)
        output = self.dropout(output)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_E(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_E, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.linear1 = torch.nn.Linear(cnn_out_shape[-1], 1024)
        self.lstm_si1 = torch.nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout_lstm_si = nn.Dropout(p = 0.3)
        self.lstm_si2 = torch.nn.LSTM(input_size=512, hidden_size=256, num_layers=2, bidirectional=False, batch_first=True)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(p = 0.4)
        self.dropout2 = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        feature_maps = self.dropout1(feature_maps)
        fc = self.linear1(feature_maps)

        out, (_, _) = self.lstm_si1(fc)
        out = self.dropout_lstm_si(out)
        out, (_, _) = self.lstm_si2(out)

        out = self.linear2(out) 
        output = self.linear3(out)
        output = self.dropout2(output)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    
class LRCN_F(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_F, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_bi_do = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_bi_do(feature_maps)
        out = self.linear(out)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        #output = output.permute(0, 2, 1)
        #return out, output
        return out, output.view(output.shape[0], output.shape[2], output.shape[1])
        #return output.view(output.shape[0], output.shape[2], output.shape[1])

class LRCN_F1(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_F1, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_bi_do = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=2) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_bi_do(feature_maps)
        out = self.linear(out)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        output = output.permute(0, 2, 1)
        return out, output
        #return out, output.view(output.shape[0], output.shape[2], output.shape[1])
        #return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_G(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_G, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_bi_do = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.lstm_bi = torch.nn.LSTM(input_size=1024, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        out, (_, _) = self.lstm_bi_do(feature_maps)
        out, (_, _) = self.lstm_bi(out)
        out = self.linear(out)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_H(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_H, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.linear1 = nn.Linear(cnn_out_shape[-1], 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm_bi_do = nn.LSTM(input_size=1024, hidden_size=512, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        feature_maps = self.dropout1(feature_maps)
        feature_maps = self.linear1(feature_maps)
        out, (_, _) = self.lstm_bi_do(feature_maps)
        out = self.linear2(out)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

class LRCN_I(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_I, self).__init__()
        self.cnn_block = optimal_cnn_block
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.linear1 = nn.Linear(cnn_out_shape[-1], 1024)
        self.dropout1 = nn.Dropout(0.4)
        self.lstm_bi_do = nn.LSTM(input_size=1024, hidden_size=512, num_layers=3, dropout=0.3, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear3 = nn.Linear(512, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1) 

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        feature_maps = self.dropout1(feature_maps)
        feature_maps = self.linear1(feature_maps)
        out, (_, _) = self.lstm_bi_do(feature_maps)
        out = self.linear2(out)
        output = self.dropout(out)
        output = self.linear3(output)
        output = self.dropout(output)
        output = self.logSoftmax(output)
        return output.view(output.shape[0], output.shape[2], output.shape[1])

class LRCN_FA(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_FA, self).__init__()
        self.cnn_block = optimal_cnn_block
        self.hidden_size = 512
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_bi_do = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=2) 
        self.w_s1 = nn.Linear(2 * self.hidden_size, 350)
        self.w_s2 = nn.Linear(350, 5)
        self.fc_layer = nn.Linear(30 * 2 * self.hidden_size, 2000)
        self.label = nn.Linear(2000, num_classes)

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.w_s2(F.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix

    def forward(self, sequence):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        #print("feature_maps", feature_maps.shape)
        output, (h1, h2) = self.lstm_bi_do(feature_maps)
        #output = out.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        #fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        #out = self.label(fc_out)

        #print("hidden_matrix", hidden_matrix.shape)
        out = self.linear(hidden_matrix)
        #out += probability
        #print("out", out.shape)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        #print("output", output.shape)
        output = output.permute(0, 2, 1)
        return out, output
        #return out, output.view(output.shape[0], output.shape[2], output.shape[1])

class LRCN_FA_Probability(nn.Module):
    def __init__(self, optimal_cnn_block, num_channels, segment_size, num_classes):
        super(LRCN_FA_Probability, self).__init__()
        self.cnn_block = optimal_cnn_block
        self.hidden_size = 512
        cnn_out_shape = self.cnn_block(torch.rand(1, num_channels, segment_size)).shape
        self.lstm_bi_do = torch.nn.LSTM(input_size=cnn_out_shape[-1], hidden_size=512, num_layers=2, dropout = 0.3, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=2) 
        self.w_s1 = nn.Linear(2 * self.hidden_size, 350)
        self.w_s2 = nn.Linear(350, 5)
        self.fc_layer = nn.Linear(30 * 2 * self.hidden_size, 2000)
        self.label = nn.Linear(2000, num_classes)
        self.weight = torch.nn.Parameter(data=torch.randn(4, num_classes), requires_grad=True)
                


    def attention_net(self, lstm_output):
        attn_weight_matrix = self.w_s2(F.tanh(self.w_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim = 2)
        return attn_weight_matrix

    def forward(self, sequence, probability):
        feature_maps = []
        for segment in sequence:
            feature_maps.append(self.cnn_block.forward(segment))
        feature_maps = torch.stack(feature_maps)
        #print("feature_maps", feature_maps.shape)
        output, (h1, h2) = self.lstm_bi_do(feature_maps)
        #output = out.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        #fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        #out = self.label(fc_out)

        #print("hidden_matrix", hidden_matrix.shape)
        out = self.linear(hidden_matrix)
        out1 = probability * self.weight
        out += torch.sum(out1, axis = 2)
        #print(probability[0][0])
        output = self.dropout(out)
        orig_out = output
        output = self.logSoftmax(output)
        #print("output", output.shape)
        #print("orig_out", orig_out[0])
        #print("output", output[0])
        #assert 1 == 2
        output = output.permute(0, 2, 1)
        return orig_out, output
        #return orig_out, output.view(output.shape[0], output.shape[2], output.shape[1])
