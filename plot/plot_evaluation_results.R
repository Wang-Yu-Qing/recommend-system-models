library('ggplot2')

df = read.csv('../evaluation_results/userCF-MovieLens.csv')
metric = c(df$recall, df$precision, df$coverage)
type = rep(c('recall', 'precision', 'coverage'), each=nrow(df))
k = rep(df$k, times=3)
n = rep(df$n, times=3)
ensure_new = rep(df$ensure_new, times=3)
IIF = rep(df$IIF, times=3)
df = data.frame(k, n, metric, type, ensure_new, IIF)
df$metric = round(df$metric, 4)
ensure_new = df[df$ensure_new=='True', ]
ggplot(ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  ylim(c(0, 0.6)) +
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-1) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=20)) +
  ggtitle("ensure new")

not_ensure_new = df[df$ensure_new=='False', ]
ggplot(not_ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  ylim(c(0, 0.5)) +
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-1) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=20)) +
  ggtitle("not ensure new")



df = read.csv('../evaluation_results/userCF-Retailrocket.csv')
metric = c(df$recall, df$precision, df$coverage)
type = rep(c('recall', 'precision', 'coverage'), each=nrow(df))
k = rep(df$k, times=3)
n = rep(df$n, times=3)
ensure_new = rep(df$ensure_new, times=3)
IIF = rep(df$IIF, times=3)
df = data.frame(k, n, metric, type, ensure_new, IIF)
ensure_new = df[df$ensure_new=='True', ]
ggplot(ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  ylim(c(0, 0.5)) +
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-1) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=20)) +
  ggtitle("ensure new")

not_ensure_new = df[df$ensure_new=='False', ]
ggplot(not_ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  ylim(c(0, 0.5)) +
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-1) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=20)) +
  ggtitle("not ensure new")