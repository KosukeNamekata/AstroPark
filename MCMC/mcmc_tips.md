# ここでは、MCMCの知識・小技をまとめていく　　


### ＊Parallel Tempering
- parallel temperingのHyperparameter(逆温度)の調整は、基本的に、Araki et al. 2013に従えばいい。  

- ただし、レプリカの数は、パラメータの数のルート(e.g.,parameter number=50, replica number=7とかでいいのか、、)が「数学的には」最適らしい。
また、初期値は、exp(0), exp(1), exp(2)...とするのがいいらしい。　　

- temperingの逆温度は、adaptiveで動かすことが多いが、an=1/(burn_in*iteration_number + n)とかにするとよい。
ここで、nは要らないように見えるが、数学的には、an(n→∞)=0, Σan(n→∞)=∞である数列の方が好ましい。  

- Emerging-Dacay Rateも、Areaと同様に、共分散にした方がいいか？→どちらでもいい。  

- prior distributionを設定する上での注意点：  
-- Emerging-Decay rate, Areaに関しては、log分布??(名前忘れた)、p(a)=1/a/(log(a_max/a_min))にすると良い。  
-- この時、提案分布も、この形にしなければならない。つまり、Q(y,x) = 1/y*exp(-( ln(y)-ln(x) )/2σ^2)  
-- その場合、メトロポリスの、次を選ぶ確率のところに、Q = 1/y*exp(-( ln(y)-ln(x) )/2σ^2)を分母分子にかけないといけないので注意。  


### ＊モデルの比較

- モデルの比較には、「**ベイズファクター**」を比較するのが良い。例えば、ベイスファクターは、B(n,n+1) = p(D|M2)/p(D|M1)として書かれる。
ここで、分母分子は、∮(Likelihood)×(Prior)dΘ(i.e.全空間積分)として計算できる。これが、「**大きいもの**」を選べば良い。
