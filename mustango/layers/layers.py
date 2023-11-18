import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import json
import yaml
class Fundamental_Music_Embedding(nn.Module):
    def __init__(self, d_model, base, if_trainable = False, if_translation_bias_trainable = True, device='cpu', type = "se",emb_nn=None,translation_bias_type = "nd"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.base = base
        self.if_trainable = if_trainable #whether the se is trainable 
        
        if translation_bias_type is not None:
            self.if_translation_bias = True
            self.if_translation_bias_trainable = if_translation_bias_trainable #default the 2d vector is trainable
            if translation_bias_type=="2d":
                translation_bias = torch.rand((1, 2), dtype = torch.float32) #Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)[0,1)
            elif translation_bias_type=="nd":
                translation_bias = torch.rand((1, self.d_model), dtype = torch.float32)
            translation_bias = nn.Parameter(translation_bias, requires_grad=True)
            self.register_parameter("translation_bias", translation_bias)
        else:
            self.if_translation_bias = False

        i = torch.arange(d_model)
        angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / d_model)
        angle_rates = angle_rates[None, ... ]#.cuda()

        if self.if_trainable:
            angles = nn.Parameter(angle_rates, requires_grad=True)
            self.register_parameter("angles", angles)
        
        else:
            self.angles = angle_rates


    def __call__(self, inp, device):
        if inp.dim()==2:
            inp = inp[..., None] #pos (batch, num_pitch, 1)
        elif inp.dim()==1:
            inp = inp[None, ..., None] #pos (1, num_pitch, 1)
        angle_rads = inp*self.angles.to(device) #(batch, num_pitch)*(1,dim)

        # apply sin to even indices in the array; 2i
        angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

        pos_encoding = angle_rads.to(torch.float32)
        if self.if_translation_bias:
            if self.translation_bias.size()[-1]!= self.d_model:
                translation_bias = self.translation_bias.repeat(1, 1,int(self.d_model/2))
            else:
                translation_bias = self.translation_bias
            pos_encoding += translation_bias
        else:
            self.translation_bias = None
        return pos_encoding
    

class Music_PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, if_index = True, if_global_timing = True, if_modulo_timing = True, device = 'cuda:0'):
        super().__init__()
        self.if_index = if_index
        self.if_global_timing = if_global_timing
        self.if_modulo_timing = if_modulo_timing
        self.dropout = nn.Dropout(p=dropout)
        self.index_embedding = Fundamental_Music_Embedding(
            d_model = d_model, base=10000, if_trainable=False, translation_bias_type = None, 
            if_translation_bias_trainable = False, type = "se"
        )# .cuda()
        self.global_time_embedding = Fundamental_Music_Embedding(
            d_model = d_model, base=10001, if_trainable=False, translation_bias_type = None, 
            if_translation_bias_trainable = False, type = "se"
        )# .cuda()
        self.modulo_time_embedding = Fundamental_Music_Embedding(
            d_model = d_model, base=10001, if_trainable=False, translation_bias_type = None, 
            if_translation_bias_trainable = False, type = "se"
        )# .cuda()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        '''
        if self.if_global_timing:
            print("pe add global time")
        if self.if_modulo_timing:
            print("pe add modulo time")
        if self.if_index:
            print("pe add idx")
        '''
    def forward(self, inp,dur_onset_cumsum = None):

        if self.if_index:
            pe_index = self.pe[:inp.size(1)] #[seq_len, batch_size, embedding_dim]
            pe_index = torch.swapaxes(pe_index, 0, 1) #[batch_size, seq_len, embedding_dim]
            inp += pe_index
        
        if self.if_global_timing:
            global_timing = dur_onset_cumsum
            global_timing_embedding = self.global_time_embedding(global_timing)
            inp += global_timing_embedding
        
        if self.if_modulo_timing:
            modulo_timing = dur_onset_cumsum%4
            modulo_timing_embedding = self.modulo_time_embedding(modulo_timing)
            inp += modulo_timing_embedding
        return self.dropout(inp)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[:x.size(1)] #[seq_len, batch_size, embedding_dim]
        pos = torch.swapaxes(pos, 0, 1) #[batch_size, seq_len, embedding_dim]
        x = x + pos
        return self.dropout(x)



class chord_tokenizer():
    def __init__(self,seq_len_chord=88,if_pad = True):

        # self.pitch_dict = {'pad': 0, "None":1, "A": 2, "A#": 3, "Bb":3, "B":4, "C":5, "C#":6, "Db":6, "D": 7, "D#":8, "Eb":8, "E": 9 ,"F":10, "F#":11, "Gb":11, "G":12, "G#":13, "Ab":13}  
        self.pitch_dict = {'pad': 0, "None":1, "N":1, "A": 2, "A#": 3, "Bb":3, "B":4, "Cb": 4, "B#":5, "C":5, "C#":6, "Db":6, "D": 7, "D#":8, "Eb":8, "E": 9 , "Fb": 9, "E#": 10, "F":10, "F#":11, "Gb":11, "G":12, "G#":13, "Ab":13}  
        self.chord_type_dict = {'pad': 0, "None": 1,"N": 1, "maj": 2, "maj7": 3, "m": 4, "m6": 5, "m7": 6, "m7b5": 7, "6": 8, "7": 9, "aug": 10, "dim":11} #, "/": 
        self.chord_inversion_dict = {'pad': 0, "None":1, "N":1,"inv": 2, "no_inv":3}
        self.seq_len_chord = seq_len_chord
        self.if_pad = if_pad

    def __call__(self, chord, chord_time):


        if len(chord)==0:
            chord, chord_time = ["N"], [0.]


        if self.if_pad:
            pad_len_chord = self.seq_len_chord - len(chord)
            chord_mask = [True]*len(chord) +[False]*pad_len_chord
            
            chord += ["pad"]*pad_len_chord
            chord_time += [chord_time[-1]]*pad_len_chord

        else:
            chord_mask = [True]*len(chord)

        self.chord_root, self.chord_type, self.chord_inv = self.tokenize_chord_lst(chord)
        self.chord_time = chord_time
        self.chord_mask = chord_mask
        # print("out",self.chord_root, self.chord_type, self.chord_inv, self.chord_time, self.chord_mask)
        return self.chord_root, self.chord_type, self.chord_inv, self.chord_time, self.chord_mask
    
    def get_chord_root_type_inversion_timestamp(self, chord):
        if chord =="pad":
            return "pad", "pad", "pad"

        if chord =="N":
            return "N", "N", "N"
        
        if len(chord.split('/'))>1:
            chord_inv = "inv"
        else:
            chord_inv = "no_inv"
        
        chord_wo_inv = chord.split('/')[0]


        if len(chord_wo_inv)>1: # this part might have a '#' or 'b'
            if chord_wo_inv[1]=='#' or chord_wo_inv[1]=='b':
                chord_root=chord_wo_inv[0:2]
            else:
                chord_root=chord_wo_inv[0]
        else:
            chord_root=chord_wo_inv[0]
        
        if len(chord_wo_inv)>len(chord_root):
            chord_type=chord_wo_inv[len(chord_root):]
        else:
            chord_type='maj'

        return chord_root, chord_type, chord_inv

    
    def tokenize_chord_lst(self, chord_lst):
        out_root = []
        out_type = []
        out_inv = []
        for chord in chord_lst:
            chord_root, chord_type, chord_inversion= self.get_chord_root_type_inversion_timestamp(chord)
            out_root.append(self.pitch_dict[chord_root])
            out_type.append(self.chord_type_dict[chord_type])
            out_inv.append(self.chord_inversion_dict[chord_inversion])
        return out_root, out_type, out_inv
    
class beat_tokenizer():
    def __init__(self,seq_len_beat=88,if_pad = True):
        self.beat_dict = {'pad': 0, "None":1, 1.: 2, 2.: 3, 3.:4, 4.:5, 5.:6, 6.:7, 7.:8}  
        self.if_pad = if_pad
        self.seq_len_beat = seq_len_beat
    def __call__(self, beat_lst):
        # beats = [[0.56, 1.1, 1.66, 2.24, 2.8, 3.36, 3.92, 4.48, 5.04, 5.6, 6.16, 6.74, 7.32, 7.9, 8.46, 9.0, 9.58], [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]]
        if self.if_pad:
            if len(beat_lst[0])==0:
                beat_mask = [False]*self.seq_len_beat
                beat_lst = [[0.]*self.seq_len_beat, ["pad"]*self.seq_len_beat]
            else:
                pad_len_beat = self.seq_len_beat - len(beat_lst[0])
                beat_mask = [True]*len(beat_lst[0]) +[False]*pad_len_beat
                beat_lst = [beat_lst[0]+[beat_lst[0][-1]]*pad_len_beat,  beat_lst[1]+["pad"]*pad_len_beat   ]

        else:
            beat_mask = [True]*len(beat_lst[0])
        self.beat = [self.beat_dict[x] for x in beat_lst[1]]
        self.beat_timing = beat_lst[0]

        return self.beat, self.beat_timing, beat_mask

# class beat_tokenizer_by_frame():
#     def __init__(self, frame_resolution = 0.01, max_len = 10):
        
#     def __call__(self, beat_lst):


# def timestamp2frame(,frame_resolution, max_len):

# def frame2timestamp(frame_resolution, man_len)



def l2_norm(a, b):
    return torch.linalg.norm(a-b,  ord = 2, dim = -1)

def rounding(x):
    return x-torch.sin(2.*math.pi*x)/(2.*math.pi)

class Chord_Embedding(nn.Module):
    def __init__(self, FME, PE, d_model = 256, d_oh_type = 12, d_oh_inv = 4):
        super().__init__()
        self.FME = FME
        self.PE = PE
        self.d_model = d_model
        self.d_oh_type = d_oh_type
        self.d_oh_inv = d_oh_inv
        self.chord_ffn = nn.Linear(d_oh_type + d_oh_inv + d_model + d_model, d_model) #.cuda()
    def __call__(self, chord_root, chord_type, chord_inv, chord_timing, device):
        #chords: (B, LEN, 4)
        #Embed root using FME
        #Embed chord type, chord inversion using OH
        #Embed timestamps using shared PE
        chord_root_emb = self.FME(chord_root, device)
        # print(chord_root_emb.size())
        # print('this is chord root: ', chord_root)
        # print('this is chord type: ', chord_type)
        # print('this is chord inv: ', chord_inv)


        # chord_root_emb = torch.randn((2,20,1024)).cuda()
        # print(chord_root_emb.device)
        # chord_root_emb = F.one_hot(chord_type.to(torch.int64), num_classes = self.d_model).to(torch.float32)
        chord_type_emb = F.one_hot(chord_type.to(torch.int64), num_classes = self.d_oh_type).to(torch.float32)
        chord_inv_emb = F.one_hot(chord_inv.to(torch.int64), num_classes = self.d_oh_inv).to(torch.float32)
        chord_time_emb = self.PE.global_time_embedding(chord_timing, device)

        chord_emb = self.chord_ffn(torch.cat((chord_root_emb, chord_type_emb, chord_inv_emb, chord_time_emb), dim = -1).to(device))
        # print("TADY toje", chord_emb.device)
        return chord_emb

        
class Beat_Embedding(nn.Module):
    def __init__(self, PE, d_model = 256, d_oh_beat_type = 4):
        super().__init__()
        self.PE = PE
        self.d_model = d_model
        self.d_oh_beat_type = d_oh_beat_type
        self.beat_ffn = nn.Linear(d_oh_beat_type+d_model, d_model)
        
    def __call__(self, beats, beats_timing, device):
        #Embed beat type using OH
        #Embed time using PE

        beat_type_emb = F.one_hot(beats.to(torch.int64), num_classes = self.d_oh_beat_type).to(torch.float32).to(device)
        beat_time_emb = self.PE.global_time_embedding(beats_timing, device)
        merged_beat = torch.cat((beat_type_emb, beat_time_emb), dim = -1)

        beat_emb = self.beat_ffn(merged_beat)
        return beat_emb

if __name__ == "__main__":
    config_path = "/data/nicolas/TANGO/config/model_embedding_config.yaml"
    with open (config_path, 'r') as f:
        cfg = yaml.safe_load(f)



    beats = [[0.56, 1.1, 1.66, 2.24, 2.8, 3.36, 3.92, 4.48, 5.04, 5.6, 6.16, 6.74, 7.32, 7.9, 8.46, 9.0, 9.58], [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]]
    beats = np.array(beats).T.tolist()
    chords = [["Gm", 0.464399092], ["Eb", 1.393197278], ["F", 3.157913832], ["Bb", 4.736870748], ["F7", 5.758548752], ["Gm", 6.501587301], ["Eb", 8.173424036], ["F7", 9.938140589]]
    
    chord_tokenizer = chord_tokenizer(seq_len_chord=30,if_pad = True)
    beat_tokenizer = beat_tokenizer(seq_len_beat=17,if_pad = True)

    #TOKENIZE CHORDS AND BEATS AT DATALOADING PART 
    chord_tokens, chord_masks = chord_tokenizer(chords)#adding batch dimension
    beat_tokens, beat_masks = beat_tokenizer(beats) 

    chord_tokens, chord_masks, beat_tokens, beat_masks = chord_tokens[None, ...], chord_masks[None, ...], beat_tokens[None, ...], beat_masks[None, ...] #adding batch dimension
    print("tokeninzing chords and beats", chord_tokens.shape, beat_tokens.shape)


    #EMBEDDING CHORDS AND BEATS WITHIN THE MODEL
    FME = Fundamental_Music_Embedding(**cfg["FME_embedding_conf"])
    PE = Music_PositionalEncoding(**cfg["Position_encoding_conf"])

    chord_embedding_layer = Chord_Embedding(FME, PE, **cfg["Chord_Embedding_conf"])
    chord_embedded = chord_embedding_layer(chord_tokens)

    beat_embedding_layer = Beat_Embedding(PE, **cfg["Beat_Embedding_conf"])
    beat_embedded = beat_embedding_layer(beat_tokens)
    print("embedding tokenized chords and beats", chord_embedded.shape, beat_embedded.shape)
