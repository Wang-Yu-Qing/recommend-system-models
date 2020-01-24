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
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-0.5) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=15)) +
  ggtitle("ensure new")+
  scale_y_continuous(breaks=NULL, limits=c(0, 0.7))


not_ensure_new = df[df$ensure_new=='False', ]
ggplot(not_ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-0.5) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=15)) +
  ggtitle("not ensure new") +
  scale_y_continuous(breaks=NULL, limits=c(0, 0.7))




df = read.csv('../evaluation_results/userCF-Retailrocket.csv')
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
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-0.5) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=15)) +
  ggtitle("ensure new") +
  scale_y_continuous(breaks=NULL, limits=c(0, 0.5))

not_ensure_new = df[df$ensure_new=='False', ]
ggplot(not_ensure_new, aes(x=type, y=metric, col=type, fill=type)) + 
  geom_bar(stat='identity', width=0.3) + 
  geom_text(aes(label=metric), vjust=-0.5) +
  facet_grid(k~IIF, labeller=label_both) +
  theme(text = element_text(size=15)) +
  ggtitle("not ensure new") +
  scale_y_continuous(breaks=NULL, limits=c(0, 0.5))




# retailrocket data set user count decription
library(dplyr)
df = read.csv('../data/Retailrocket/events.csv')
count = df %>% count(visitorid)
count1 = count[count$n<3, ]
count2 = count[(3<=count$n)&(count$n<5), ]
count3 = count[(5<=count$n)&(count$n<10), ]
count4 = count[10<=count$n, ]
count = c(nrow(count1), nrow(count2), nrow(count3), nrow(count4))
range = c('0~2', '3~4', '5~9', '9~')
ggplot(data.frame(count, range_), aes(x=range, y=count, col=range, fill=range)) +
  geom_bar(stat='identity', width=0.6) +
  theme(text = element_text(size=10)) +
  ggtitle("User frequency count for Retailrocket data set")
# retailrocket data set items count decription (long-tail)
count = df %>% count(itemid)
# sort by count
count = count[order(count$n, decreasing=TRUE), ]
count$rank = c(1:nrow(count))
ggplot(count, aes(x=rank, y=n)) +
  geom_bar(stat='identity', width=1) +
  theme(text = element_text(size=10)) +
  ggtitle("Long tail description for Retailrocket data set")



















