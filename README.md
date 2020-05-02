# covid-vis
Simple visualization of COVID deaths in different regions

This started from John Burn-Murdoch's chart that showed that country population and Covid deaths are weakly correlated.

For an explanation of the analysis, see https://twitter.com/teemu_roos/status/1254727507833741312?s=20

Model | R-squared
------|----------
population | 0.136
population & region | 0.611
population & region & Feb temp. | 0.612
population & region & pop-density | 0.628
population & region & pop-density & Feb temp. | 0.628

![](figures/Rsquared.png)
