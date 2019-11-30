import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn


class SkeletonARModel(nn.Block) :
    
    def __init__(self, **kwargs) :
        super(SkeletonARModel, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.BatchNorm(),
            rnn.LSTM(5, 3),
            nn.Dropout(0.5),
            nn.Dense(units=128, activation='relu'),
            nn.BatchNorm(),
            nn.Dense(units=64, activation='relu'),
            nn.BatchNorm(),
            nn.Dense(units=16, activation='relu'),
            nn.BatchNorm(),
            nn.Dense(units=5),
            )
    
    def forward(self, input) :
        '''
        input.shape = (N, frame 수, 34)
        '''
        X = input
        for blk in self.net :
            X = blk(X)
        return X