# fast-JL-transform
Python implementation of the Fast Johnson-Lindenstrauss transform (FJLT), as proposed in the seminal paper [The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors](https://epubs.siam.org/doi/epdf/10.1137/060673096). 


## The FJLT transform 

For a given dataset $X=\{x_1, \ldots x_N\} \subset \mathbb{R}^d$, the FJLT embeds every point into $k=O(\log N)/\epsilon^2$ dimensions as follows: 
<br>
> $y=\frac{1}{\sqrt{k}} PHD(x)$ 

where 
> $D$ is the $d\times d$ matrix with random $\pm 1$ on its diagonal, with probability $1/2$; 

> $H$ is the $d\times d$ normalized Hadamard transform

> $P$ is $k\times d$ sparse matrix, with the sparsity parameter denoted by $q$. Each entry in $P[i,j]$ is sampled as $\frac{1}{\sqrt{q}}b_{ij}N_{ij}$, where $b_{ij}$ is a Bernoulli variable with sucess probability $q$ and $N_{ij} \sim N(0, 1)$. All $b_{ij}$ and $N_{ij}$ are independent from each other.
<br>

The FJLT guarntes that if $q \sim \frac{\log^2 N}{d}$ then with high probability all the pairwise disatnces in $X$ are preserved by the FJLT within $(1+\epsilon)$ factor.

<br>

In our recent paper [The Fast Johnson-Lindenstrauss tranform is even faster](https://arxiv.org/abs/2204.01800), we analyze and improve the bound on $q$, for which the FJLT guarantee holds. Roughly, we show that setting $q \sim \frac{\log N}{d}$ sufices.  

## Work in progress:
-- write expirements to see how various values of q influence the w.c. quality of the embedding (for fixed k, N, eps, q is a function of these)
-- how q influences the average dist. quality. Compare with our Approx. In terms of speed? As well 
-- implement the Kac JL method, and play with to comapre 

