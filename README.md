## Robust Fisher Linear Discriminant Analysis (<a href="https://stanford.edu/~boyd/papers/pdf/robust_FDA.pdf">Paper</a>)

This is a collaborative project by [Chen LIU](http://liuchen1993.cn/HomePage/home.html) and [Jun LU](http://lujunzju.github.io/)

We use two datasets, both of which are from UCI. They are sonar(<a href="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data">LINK</a>) and ionosphere(<a href="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data">LINK</a>)

In addition, in order to solve the QCQP problem raised by robust LDA, we take advantage of the python library cvxopt and scipt writtern by <a href="http://pages.cs.wisc.edu/~kline/qcqp/qcqprel_py">Jeffery Kline</a>

## Code Structure
```
|-load.py            # load datas
|-FisherLDA.py       # general fisher LDA
|-robustFisherLDA.py # robust fisher LDA 
|-mainTest.py        # test on general fisher LDA and robust fisher LDA
|-log.py  # if you do not want to see so many log infos, please change the DEBUG to False
|-util.py # util functions
|-requirements.txt   # contains the requried packages
```
