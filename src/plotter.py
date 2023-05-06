
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

def viz_acc_eps(df, plt, dataset, std, epss=None, y_title=None, x_label='', lw=3, legend=None,ylim=None,):
    if epss is None:
        epss = np.sort(df.eps.unique())
        epss = epss[epss>=0]
        
    metrics = {
        'score_dl1' : r"$\hat{\Delta}_1$",
        'score_dl2' : r"$\hat{\Delta}_2$",
        'score_de1' : r"$\hat{\Delta}_{\mathsf{E}_1}$",
        'score_da'  : r"$\hat{\Delta}_{\mathcal{A}}$",
        'score_dx'  : r"$\hat{\Delta}_{\mathcal{X}}$",
    }
    lss = [None, '--', '-.', '-.', '-.']
    dt = df[df.dataset_name == dataset]
    dtt = dt[dt.eps.isin(epss)]
    for kidx, (score_key, score_lb) in enumerate(metrics.items()):            
        if '_dl' not in score_key:
            accs = dt[dt.score==score_key].groupby(['eps'])['acc'].max().values
            accs = [accs[0]] * len(epss)
            xs = epss
        else:
            accs = dtt[dtt.score==score_key].groupby(['eps'])['acc'].max().values
            xs = np.sort(dtt[dtt.score==score_key].eps.unique())

        plt.plot(xs, accs, label=score_lb, lw=lw, ls=lss[kidx])

    plt.plot(epss, [0.5]*len(epss), label=r"RD", lw=lw, ls=':')
    plt.fill_between(epss, 
                     [0.5-std]*len(epss),
                     [0.5+std]*len(epss),
                     alpha=0.2,
                     facecolor='C5',
                     edgecolor='C5',
                    )

    plt.xlabel(r"$\epsilon$"+x_label, fontsize=20)
    if y_title is not None:
        plt.ylabel(y_title, fontsize=20)

    if ylim is not None:
        plt.ylim(ylim)
        
    if legend is not None:
        plt.legend(fontsize=20, loc=legend,
               ncol=3
              ).get_title().set_fontsize(22)
    plt.gca().tick_params(axis='both', which='major', labelsize=18)
    plt.gca().set_yticks(plt.gca().get_yticks()[::2])
    plt.tight_layout()



def viz_acc_n(df, plt, dataset, score, epss=None, y_title=None, x_label='', lw=3, 
              ls='-',legend=None,ylim=None,ncol=3):
    if epss is None:
        epss = np.sort(df.eps.unique())
        epss = epss[epss>=0]
        
    metrics = {
        'score_dl1' : r"$\hat{\Delta}_1$",
        'score_dl2' : r"$\hat{\Delta}_2$",
        'score_de1' : r"$\hat{\Delta}_{\mathsf{E}_1}$",
        'score_da'  : r"$\hat{\Delta}_{\mathcal{A}}$",
        'score_dx'  : r"$\hat{\Delta}_{\mathcal{X}}$",
    }
    dt = df[(df.dataset_name == dataset) & (df.score==score)]

    for eps_id, eps in enumerate(epss):
        dtt = dt[dt.eps==eps]
        Ns = dtt.N.unique()
        grp = dtt.groupby('N').acc
        mus = grp.mean().values
        ci = np.array(grp.apply(lambda s : sms.DescrStatsW(s.values).tconfint_mean()).tolist())
        
        plt.errorbar(Ns, mus, yerr=np.abs(mus-ci.T) ,fmt='-o', ls=ls,
                     label=rf"${eps}$", c=f'C{eps_id}')
        
    plt.xlabel(rf"$N$"+x_label, fontsize=20)
    if y_title is not None:
        plt.ylabel(y_title, fontsize=20)
#         plt.ylim(0.53, 0.65)

    if ylim is not None:
        plt.ylim(ylim)
    plt.gca().tick_params(axis='both', which='major', labelsize=18)
    plt.yticks([0.5,0.55,0.6,0.65,0.7])
    plt.xticks([5,20,40,60,80,100])
    if legend is not None:
        plt.legend(title=r"$\epsilon$", fontsize=15, 
           #loc=2,
           loc=legend,
           ncol=ncol).get_title().set_fontsize(22)
    
    plt.tight_layout()
        
        