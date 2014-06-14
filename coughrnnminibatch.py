from rnn_minibatch import *
import numpy as np
import re
from rnn_util import *

def testRNN():
# Data preparation
    trainlist = '/home/james/rnn2/examples/cough_example2/train.txt'
    vallist = '/home/james/rnn2/examples/cough_example2/valid.txt'
    testlist = '/home/james/rnn2/examples/cough_example2/test.txt'
    seq = loadmfc(trainlist)
    print type(seq)
    print seq.shape
    targets = loadlabel(trainlist)
    print 'loading DONE'
    testseq = loadmfc(testlist)
    (testtargets,testfiles) = loadlabel(testlist)

    loggingFilename = '/home/james/rnn2/examples/cough_example2/rnnlogger.txt'
    logging.basicConfig(filename = loggingFilename,level=logging.INFO)
    t0 = time.time()
# Model build
    n_in = 13
    n_hidden = 120
    n_out=2
    n_epochs = 100
    n_seq = 100  # per batch
    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.005, learning_rate_decay=0.999,
                    n_epochs=n_epochs, batch_size=n_seq, activation='sigmoid',
                    output_type='binary') # from rnn_minibatch module
    model.fit(seq, targets, validate_every=10, compute_zero_one=True,
              optimizer='cg')

    #model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    #learning_rate=0.001, learning_rate_decay=0.999,
                    #n_epochs=40, activation='sigmoid') # from rnn module

    #model.fit(seq, targets, validation_frequency=100)
    model.save('/home/james/data/rnn/')
    print "Elapsed time: %f" % (time.time() - t0)
# Model test`
    seqs = xrange(len(testseq))
    for seq_num in seqs:
        guess = model.predict_proba(testseq[:, seq_num,: ])
        savelabel(guess,'/home/james/data/rnn/prelab',testfiles)


if __name__ == "__main__":
    #testlist = '/home/james/rnn2/examples/cough_example2/test.txt'
    #dd = readmfc('/home/james/data/rnn/segData/121222_001_01_20.mft')
    #dd = loadmfc(testlist)

    #dd = readlab('/home/james/data/rnn/segData/121222_001_01_20.PHN')
    #dd = loadlabel(testlist)
    testRNN()
    #print dd
    #print dd.shape
