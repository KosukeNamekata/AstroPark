# MCMC Based Bayesian Data Analysis  

## 1. What is Bayesian Statistics with MCMC?
 
- 「**ベイズ推定**」(Jeffreys 1939)と「**MCMC**」(Metropolis et al. 1953)は、二つの独立した手法で、前者は、観測データを理解する手法、後者は、サンプリングする手法のことである。  

- 「**ベイス推定**」とは、下記のベイズ確率の考え方に基づき、観測事象（観測された事実）から、推定したい事柄（それの起因である原因事象）を、確率的な意味で推論することを指す。統計学に応用されて**ベイズ統計学**の代表的な方法となっている。  

  - 標語的には、「真値は分布する」、「点推定にはこだわらない」などの考え方に依拠している。  

  - ベイズ推定に対する批判としては、事前確率が主観的で一意的に決められない、またそれをもとにして事後確率を求めても、それが客観的な確率分布に収束するという保証がない、といったものがある。  

  - しかし現在では特にコンピュータを用いた方法の発展によりベイズ推定の方法も発展し、スパムメールを識別するためのベイジアンフィルタなどの応用が進んでいる。事前分布としては全く情報がない場合には一様分布などが用いられ（もちろん情報があれば他の分布でよい）、**一般には異なる事前確率分布からマルコフ連鎖モンテカルロ法などで安定した結果（事後確率分布）が得られれば、実用的に問題はない**と考えられている。  

- 「**マルコフ連鎖モンテカルロ法**」は、求める確率分布を均衡分布として持つマルコフ連鎖を作成することをもとに、確率分布のサンプリングを行うアルゴリズムの総称である。メトロポリス・ヘイスティングアルゴリズムやギブスサンプリングなどのランダムウォーク法もこれに含まれる。充分に多くの回数の試行を行った後のマルコフ連鎖の状態は求める目標分布の標本として用いられる。試行の回数を増やすとともにサンプルの品質も向上する。

  - MCMCによるサンプリング結果は、事後分布に対応するため、ベイズ推定と相性が非常によい。  

- ベイズの事後分布において最も高い点（map推定値）を求めるということは、尤度が最も高い点を見つけるということ、つまり、MCMCで最適解を求めるということとイコール。

- 度たび、bayesian analysisは、proposal distributionの決定に困難があり、**これは現在でももっともよく研究されていて、需要が高い分野である**が、未だに確固たる王道の理論があるわけではない。  

- 事前情報が全くない場合は、一様分布を用いる(see, Kass & Wasserman 1996, for review)。priorの決定には、the Jeffreys ruleというものがある。

- **詳細釣り合いの法則(condition of detail balance)**; Marcov chainは、可逆性(reversibitity)を必ずしも必要としないが、詳細釣り合いの法則を満たせば、必ず定常状態が存在することを保証する。  

```
π(Xn)K(Xn;Xn+1) = π(Xn+1)K(Xn+1;Xn)
```



## 2. MCMCの手法  
- メトロポリス・ヘイスティング(MH、とも):　**直接サンプリングするのが難しい確率分布から統計標本の配列を生成するのに用いられるマルコフ連鎖を構築するのに用いられる手法の総称**。この配列はマルコフ連鎖モンテカルロ法において、目標分布の近似（ヒストグラム）として用いられたり、期待値のような積分計算を必要とするものに用いられる。

- **メトロポリス法**:  
<kbd><img src="https://user-images.githubusercontent.com/44762667/48882516-161aed00-ee5e-11e8-9ba5-bf181b5222ec.png" width="600px"></kbd>

  - ここで、選択率の正式表記は、`K(x; y) = min(1, f(y)q(x|y)/f(x)q(y|x))`となる。この表記は、詳細釣り合いの法則から簡単に導くことができる。(というよりも、可逆性を保つために、この表記にしている、といったほうが正確か。)

  - Decreasing σ (in proposal distribution) increases α (=acceptance ratio) but lowers the independence of the sampler, and vice versa.

  - もともとは、メトロポリスによって、統計力学の平行問題を解くことに使われ、統計の分野で広がり始めたのが始め。
統計力学では、その状態をとる確率がexp(-H/kβ)で与えられ、これをLikelihoodにして、MCMCで定常状態を探したのが、始めらしい。これは面白い。
ただし、これがあらゆる確率分布を持つ者に適用可能であるということが発見されるのは、だいぶ後のことであった(Hastings 1970)。  

  - **EMアルゴリズム**: 反復法の一種であり、期待値(英: expectation, E) ステップと最大化 (英: maximization, M)ステップを交互に繰り替えすことで計算が進行する。Eステップでは、現在推定されている潜在変数の分布に基づいて、モデルの尤度の期待値を計算する。Mステップでは、E ステップで求まった尤度の期待値を最大化するようなパラメータを求める。M ステップで求まったパラメータは、次の E ステップで使われる潜在変数の分布を決定するために用いられる。  

- **ギブス・サンプラー法**(Geman & Geman 1984, firstly used for image restoration): とあるパラメータが「尤度に対して」簡易関数(ガウス関数、など)で表せる場合、次のステップをその関数で簡単に表してしまって、必ず採択させるようにし、計算を簡単にさせる手法。特にモデル関数が初等関数であらわせる場合に用いれる。もちろん、他の手法と組み合わせることも可能。

  - Gibbs samplingは、複数のステップに分けられる。それぞれのステップにて、一つ一つパラメータをアップデートする。

- **Affine invariant sampling　/ ensemble sampler　(Gilks, Roberts & George 1994, Goodman & Weare 2010)**  

  - multiple chains (walkers) are run in parallel but allowed to interact in such a way that they can adapt their proposal densities. (Parallel temperingと似ているが、　proposal densityのみをやりとり捨という点で、異なる。)　　
  
  - A python implementation of this (emcee: the MCMC hammer, http://dan.iel.fm/emcee/current/) is provided by Foreman-Mackey et al. (2013) and is widely used in astronomy.  　　
  
  - 

- 他にも、スライスサンプリング, ハイブリッドモンテカルロ, MTM アルゴリズム

- **Hamiltonian・モンテカルロ法**: 「Stan」初速度を与えて、ブラーンと力学運動させる。それを繰り返すことで、効率よくマッピングするらしい。  

  - 「**Stan**」はpythonにもRにも施されている、**もっとも開発が暑いプログラミングソフト(パッケージ)**。尤度関数と事前分布を与えるだけで、事後分布をサンプリングしてくれるという便利なソフト。(1)Hamiltonian MCMCも実装+(2)変分ベイズ、が特徴。

### ＊提案分布について、
- 相関の強いパラメータがある場合は共分散を入れるのは必須。


### ＊Why Bayesian Statistics with MCMC?  
- なぜMCMCを使うのか？  
  - パラメータ次元が多すぎて、カイ2乗推定では、サーベイするパラメータレンジが膨大すぎて、計算が爆発してしまう。
かといって、勾配降下法では、高確率で、極所解に陥ってしまう(さらにエラーももとまらない)。**MCMCだと（完全ではないものの）広い空間を限られた時間で探索できる**。  

### ＊今回とる手法: **Metropolis MCMC with Adaptive Algorithm and Parallel Tempering**  
<kbd><img src="https://user-images.githubusercontent.com/44762667/48882518-1dda9180-ee5e-11e8-8ea9-16b3043525ac.png" width="600px"></kbd>

  - MCMCの採択確率は25%、交換確率は50%になるように提案分布の分散パラメータや逆温度パラメータを設定するのが適切?


### ＊Parallel Tempering
- 交換頻度は経験的に10-20%くらいが適切。交換頻度が大きすぎると交換ばかり起こってパラメータが推移しなくなり、逆に頻度が小さくなると局所解に長い間トラップされるようになります。  

- parallel temperingのHyperparameter(逆温度)の調整は、基本的に、Araki et al. 2013に従えばいい。ただし、レプリカの数は、パラメータの数のルート(e.g.,parameter number=50, replica number=7とかでいいのか、、)が「数学的には」最適らしい。　また、初期値は、exp(0), exp(1), exp(2)...とするのがいいらしい。　　

- 提案分布の分散の初期値は本来予想される値よりも大きめの値からスタートするのが良い、と聞いたことがある。学習して適切な値まで小さくなって行くので、最初から小さい値にすると学習の効果がない。逆温度の初期値に関しては、最小の値の設定がまずは重要。パラメータ空間を十分自由に動けるような値で最小の逆温度の値を設定すべきで、具体的な値は試行錯誤するしかないと思います。逆温度は一般的には「対数スケールで等間隔」に設定するのが良いそうだが、特に相転移が起きそうな温度付近は密に刻むべき。  

- temperingの逆温度は、adaptiveで動かすことが多いが、an=1/(burn_in*iteration_number + n)とかにするとよい。
ここで、nは要らないように見えるが、数学的には、an(n→∞)=0, Σan(n→∞)=∞である数列の方が好ましい。
ただし、adaptiveに学習する際の、数列係数の設定は、**非常に悩ましい**。色々試行錯誤して、今は最初からかなり小さな係数にして、正解に近いパラメータから始めて、時間をかけて学習することで良く機能する。

- Emerging-Dacay Rateも、Areaと同様に、共分散にした方がいいか？→どちらでもいい。  

- prior distributionを設定する上での注意点：  
  - Emerging-Decay rate, Areaに関しては、log分布??(名前忘れた)、p(a)=1/a/(log(a_max/a_min))にすると良い。  
  - この時、提案分布も、この形にしなければならない。つまり、Q(y,x) = 1/y*exp(-( ln(y)-ln(x) )/2σ^2)  
  - その場合、メトロポリスの、次を選ぶ確率のところに、Q = 1/y*exp(-( ln(y)-ln(x) )/2σ^2)を分母分子にかけないといけないので注意。  


## 3. モデルの比較

-  **ベイズファクター**(for a review and a guide to interpreting the Bayes factor, see Kass & Raftery 1995)。  

  - ベイスファクターは、B(n,n+1) = p(D|M2)/p(D|M1)として書かれる。
ここで、分母分子は、∮(Likelihood)×(Prior)dΘ(i.e.全空間積分)として計算できる。これが、「**大きいもの**」を選べば良い。　　
https://ja.wikipedia.org/wiki/%E3%83%99%E3%82%A4%E3%82%BA%E5%9B%A0%E5%AD%90　　

    - ベイズ因子（ベイズいんし、英: Bayes factor）は、ベイズ統計学において、伝統的統計学の仮説検定に代わる方法として用いられる数値である。
データベクトルx に基づいて2つの数学的モデル M1 と M2 のどちらかを選択する問題を考える。
この方法は尤度比検定あるいは最尤法に似ているが、尤度（モデルあるいは母数を定数とし、それを条件とする確率変数x の条件付確率のこと）を最大化するのでなく、母数を確率変数とし、それに対して平均値をとってから最大化するところが違う。

    - 基準は以下より、、  
https://www.stat.washington.edu/raftery/Research/PDF/kass1995.pdf　　  
簡単な例(広島カープ)を使って解説　　  
https://www.slideshare.net/kazutantan/bayes-factor　　

    - ただし、ほとんどの場合、ベイズファクターを計算するのは、困難である。よって、BIC条件などが基準んい使われる。これを最小化するように、パラメータを選べば良い。パラメータで記述されたモデルのクラスからモデルを選択する基準．Schwarz情報量規準とも呼ばれる． k 個のパラメータをもつ分布 f(x|θ) に従って N 個のデータがサンプルされているとき，次式を最大化するモデルを選択する。ただし，Pr[{x}N|θ] は尤度。(参考　http://ibisforest.org/index.php?BIC　)  
```
BIC=−2log(Pr[{x}N|θ])+klogN  
```


- **Predictive methods**

  - Bayesian methodも、predictive methodも、利点と欠点の両方を備えている。もし、事前分布がよく正当化されているのであれば、bayesian factorを比較するのが良い。一方、もし我々の目的が、将来のデータに対する正確な予言であるならば、predictive methodがベターであろう。　　

## 4. MCMCの収束判定(Convergence, i.e., how long should we run an MCMC chain?, see Cowles & Carlin (1996) for review)    

- 残念ながら、収束性を検出する手法はないが、**収束失敗しているのを判定することはできる**。収束判定は必要条件ではあるが、十分条件ではない。  

1. *Effective sample size*
- Autocorrelation fucntionを見てみると、correlated distributionに対しては、exp(-t/t_x)となる。
この、Autocorrelation functionを、全空間積分することで、得られる*T*を用いて、**ESS(Effective Sample Size)=N/2Tが、1000<ESS<4000程度になるのが理想**（see, e.g. Sokal 1997)。  

2. *Variance between chains*...Gelman-Robinの収束判定法  
- 最も広く使われている方法




## Appendix...関連文献  

- パラレルテンパリングをAdaptiveで入れた研究(Araki et al. 2013)  
https://www.sciencedirect.com/science/article/pii/S0893608013000415  

- Radial Velocityをモデリングする論文  
http://adsabs.harvard.edu/abs/2005ApJ...631.1198G  
http://adsabs.harvard.edu/abs/2005AJ....129.1706F  
http://adsabs.harvard.edu/abs/2006ApJ...642..505F  

- Bayesianモデリングの本  
http://adsabs.harvard.edu/abs/2005blda.book.....G  

- MCMC全体のReview論文  
http://adsabs.harvard.edu/abs/2017ARA%26A..55..213S  

- MCMCによる黒点モデリング論文  
http://adsabs.harvard.edu/abs/2006PASP..118.1351C    

> 読んだ感じ、一般に使えるコードを開発しているものの、未だにあまり知られていない？同様に、パラメータに縮退があることは述べられている。ただし、時間変化は解いていないので、そこまですごい研究ではなさそう。


- MCMCのオープンコード(Corner plotが簡単に書ける)、十数パラメータ位ならこちらで可能か。  　　
http://adsabs.harvard.edu/abs/2013PASP..125..306F  
> 天文学者が開発したので、この分野には応用がしやすいかもしれない。ただ、少し使った感じ、範囲指定とかは少し厄介か、という印象。加えて、KMRさんは、これはあまり効率が良くないと言っていた。(初期条件をいっぱいおくことで、初期条件依存性などが解消されて結構いいのではないかと思うのだけども。ただし、パラメータが多くなるとそうもいかない。)　　

- 解析的な黒点モデル論文    
http://adsabs.harvard.edu/abs/2012MNRAS.427.2487K  

- 黒点パラメータの縮退を議論した論文  
http://adsabs.harvard.edu/abs/2013ApJS..205...17W　　　　  
> 確かに、星のインクリネーションと緯度情報は、「厳密にではないが、エラーの範囲で」縮退する。この論文では、黒点の温度・面積の縮退は述べられてはいるが、定量的に評価はなされていない。黒点面積・温度・緯度・インクリネーションの縮退がどれほどあるか？ケプラーの測光精度ではどれくらい分解できるか？といった評価は、自分ですべきであろう。この自分の評価を正当化させるためにも。

- 山田さん(植村さんの学生さん)のadaptive MCMCのproceeding(論文は準備中だそうです)　　
http://adsabs.harvard.edu/abs/2017ifs..confE..30Y　　  
> この論文では、adaptiveの数列を、1/(100+n)遠いているが、これがどうやって決められたのかは気になるところ。  

## ＊疑問点・勉強すべきこと・研究コメント  

- 共分散行列を入れるとはどういうことか？　　  

- 提案分布を変更した時に、どのようにメトロポリスに反映させればいいか？  

- 細かいところの引用文献  

- tempering temperatureよりも、standard deviationの方が、早めに動いた方がいいかもしれない。というか、全体的に大きく動くようにした方がいいのかもしれない。ただ、全体的に早く動かすと、temperingで動きにくくなるのではないかと思う。やっぱり、レプリカの数は大きくした方がいいのかもしれない。

- 収束パラメータがよくわからない?

- 変更すべきパラメータ:  
(1)**標準偏差in提案分布**, (2)**Adaptiveの数列×2**, (3)**黒点の数**  
(3)は、Bayes Factorを計算して対処。これは一番最後にすればいい。
(1)は、少し広めにとるのが、いい。そして、adaptiveで調整していくのがいいでしょう。
(2)が一番厄介かもしれない。burn_inの内である程度修正できれば十分。
なので、exp(-t/burn_in_sample)×a_n, a_n=1/(burn_in_sample+n)とかの方がいいかも。


