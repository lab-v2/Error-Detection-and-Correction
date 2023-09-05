import torch
import torch.nn as nn
from torch.nn import functional as F


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
        return output.view(output.shape[0], output.shape[2], output.shape[1])
    

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
        self.logSoftmax = nn.LogSoftmax(dim=1) 
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
        #print("out", out.shape)
        output = self.dropout(out)
        output = self.logSoftmax(output)
        #print("output", output.shape)
        return output.view(output.shape[0], output.shape[2], output.shape[1])
