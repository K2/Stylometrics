Stylometric Detection of AI-Generated Text in
Twitter Timelines
Tharindu Kumarage1, Joshua Garland1, Amrita Bhattacharjee1, Kirill
Trapeznikov2, Scott Ruston1, and Huan Liu1
1 Arizona State University, Tempe AZ, USA
2 STR, Woburn MA, USA
{kskumara,jtgarlan,abhatt43,sruston,huanliu}@asu.edu1
kirill.trapeznikov@str.us2
Abstract. Recent advancements in pre-trained language models have
enabled convenient methods for generating human-like text at a large
scale. Though these generation capabilities hold great potential for break-
through applications, it can also be a tool for an adversary to generate
misinformation. In particular, social media platforms like Twitter are
highly susceptible to AI-generated misinformation. A potential threat
scenario is when an adversary hijacks a credible user account and in-
corporates a natural language generator to generate misinformation.
Such threats necessitate automated detectors for AI-generated tweets
in a given user’s Twitter timeline. However, tweets are inherently short,
thus making it difficult for current state-of-the-art pre-trained language
model-based detectors to accurately detect at what point the AI starts
to generate tweets in a given Twitter timeline. In this paper, we present
a novel algorithm using stylometric signals to aid detecting AI-generated
tweets. We propose models corresponding to quantifying stylistic changes
in human and AI tweets in two related tasks: Task 1 - discriminate be-
tween human and AI-generated tweets, and Task 2 - detect if and when
an AI starts to generate tweets in a given Twitter timeline. Our exten-
sive experiments demonstrate that the stylometric features are effective
in augmenting the state-of-the-art AI-generated text detectors.
Keywords: AI generated text · Large language models · Twitter · Sty-
lometry· Misinformation
1 Introduction
With the recent advances in transformer-based language models, we see tremen-
dous improvements in natural language generation (NLG). Consequently, with
the proliferation of pre-trained language models (PLM) such as Grover [29],
GPT-2 [23] and GPT-3 [18] the generation of human-like texts by AIs, i.e.,
AI-generated text, has become easy and achievable at large-scale. A research
question that emerges with the advancement of NLG is: can AI-generated text
be automatically detected? This is primarily because NLG models can generate
grammatically accurate large volumes of text backed by the pre-trained language
arXiv:2303.03697v1 [cs.CL] 7 Mar 2023
2 Kumarage et al.
models, thus making way for potential societal and ethical issues. An adversary
could incorporate these models with malicious intent and produce text that could
lead to harm and confusion. Some examples such as click-bait headlines [20],
deep tweets [5], and AI-generated fake news [29] show the underlying potential
threats.Change Point
Human-Written
Tweets
Machine-Generated
Tweets
Fig. 1: An hypothetical example where a credible
news Twitter account gets hijacked and gener-
ates misinformation.
Social networks such as
Twitter are an ideal play-
ground for adversaries to in-
corporate such AI text gen-
erators to generate misinfor-
mation on a large scale. For
example, an adversary could
deploy bots on social me-
dia, equipped with text gen-
erators to disseminate AI-
generated misinformation or
even launch large-scale misin-
formation campaigns. In this
paper, we consider a new
threat scenario as shown in
Fig. 1, depicting a Twitter
timeline of a credible user account and how the author changes from a human
(credible user) to an AI (NLG). Here an authentic Twitter account gets hacked
by a malicious user who then incorporates an AI text generator to generate
misinformation. The severity of these types of malicious human-to-AI author
changes is twofold: 1) credible accounts have a vast number of followers, hence
a high diffusion rate of misinformation, and 2) compelling human-like AI tweets
are generated at an unprecedented pace. Therefore, to identify this threat, it is
crucial to have an automatic mechanism for detecting AI-generated tweets on
Twitter. Furthermore, detecting the point where the human-to-AI author change
occurs would be vital for digital forensics and future threat mitigation.
Many approaches exist for automatically detecting AI-generated text in the
literature, the most successful of which use PLMs [5,29], However, incorporating
the state-of-the-art (SOTA) PLM-based classifiers for detecting human-to-AI
author changes in Twitter timelines is particularly challenging for two reasons:
1) Input text contains fewer semantic information. Tweets are inherently
short in length, and the classification accuracy of PLMs decreases when the input
text length is small and the amount of semantic information is insufficient.
2) Generating training data for supervised learning. PLM-based clas-
sifiers require sufficient fine-tuning to adjust to the task at hand. The training
data in this problem would consist of Twitter timelines which each contain a
sequence of human and AI-generated tweets. It is a resource-consuming task to
generate such AI-generated Twitter timelines. To address the challenges, we pro-
pose a simple yet effective architecture using stylometric features as an auxiliary
signal to detect AI tweets in Twitter timelines. Accordingly, we analyze different
categories of stylometric features and design a comprehensive set of experiments
Stylometric Detection of AI-Generated Text in Twitter Timelines 3
to discuss how stylometry augments AI-generated text detection performance
across various configurations, e.g., AI-generator size, tweet topic, and others.
Furthermore, we propose a simple stylometric feature-based change-point detec-
tion architecture to detect if and when a human-to-AI author change occurs in
a user’s timeline. This methodology consists of few learnable parameters and
works well even when only a few training timeline samples exist. To summarize,
we study the following two research questions:
RQ1: When detecting AI-generated tweets from a timeline, can stylometric
features improve the performance of SOTA text detectors?
RQ2: With limited training data, how well can stylometric features detect
if and when a human-to-AI author change occurs in a user’s Twitter timeline?
We evaluate our stylometry architectures on two datasets3: an in-house dataset
created to emulate the human-to-AI author change in a user’s Twitter timeline
and a publicly available dataset, TweepFake [5]. Our results on both datasets
empirically show that 1) stylometric features improve existing PLM-based AI-
generated text classifiers significantly when the length of the Twitter timeline is
small, and 2) stylometric signals help detect when an author change occurs in a
Twitter timeline, mainly when there is limited training data.
2 Related Work
Bot Detection on Twitter. There exists a large body of work on bot detection
methods on Twitter. Most bot detection methods use user account features (such
as follower counts, likes, retweets, etc.) or temporal features such as activity
[4, 6, 15, 22]. Unlike standard bot detection methods, our objective is to purely
use the raw text of the Tweet sequence and identify whether a language model
generates the text in the sequence. Hence, we do not compare our method with
bot detection baselines and instead focus on AI-generated text detection work.
AI-Generated Text Detection. Initial research on generated text detection
incorporated techniques such as bag-of-word and tf–idf encoding followed by
standard classifiers such as logistic regression, random forest, and SVC [10]. In
recent years [29] showed the effect of exposure bias on detecting text generated
by large language models. Consequently, the subsequent works used pre-trained
language model architectures (BERT, RoBERTa, GPT-2, etc.) as the detector
and showed state-of-the-art results in detecting AI-generated text in many do-
mains [8, 19, 21, 24]. Similarly, a finetuned RoBERTa-based detector has also
shown significant performance in detecting AI-generated tweets [5]. Few recent
works in generated text detection further attempt to extend the PLM-based de-
tectors with new learning paradigms such as Energy-Based Model(EBM) [1] and
additional information such as the factual structure and topology [16,30]. In one
of the recent works, the authors incorporated text augmentation to improve the
performance of detecting AI-generated text in Twitter [26].
3 Our detection code is available at https://github.com/TSKumarage/Stylo-Det-AI-
Gen-Twitter-Timelines.git
4 Kumarage et al.
Stylometry for AI-Generated Text Detection. It has been shown by Schus-
ter et al. [19] that stylometry has limited usage when trying to discriminate be-
tween AI-generated real news and AI-generated fake news. In contrast, our goal
in this paper is to use stylometry to discriminate between human-written text
and AI-generated text. To our knowledge, this is the first work incorporating sty-
lometry to discriminate AI-generated text from human-written text. However,
stylometry is a well-established tool used in author attribution and verification
in many domains, including Twitter [2]. For detecting style changes within a
document, different stylistic cues are leveraged in order to identify a given text’s
authorship and find author changes in multi-authored documents [9, 28]. Our
work differs from these in that, while they detect human author changes within
multi-authored documents, we measure human-to-AI author changes within a
given Twitter timeline. However, the underlying hypothesis is similar. PAN [28]
is a series of scientific events and shared tasks on digital text forensics and sty-
lometry. In past years, PAN has examined multiple applications of stylometry
for the detection of style changes in multi-authored documents. A couple of no-
table examples are a BERT-based model [11], and an ensemble approach which
incorporates stylometric features [25]. We use these two models as baselines in
our study.
3 Preliminaries
To address the research questions in our study, we formulate the following two
tasks: 1) Human vs. AI Tweet Detection and 2) Human-to-AI Author Change
Detection and Localization. We formally define these tasks as follows:
1) Human- vs. AI-Authored Tweet detection. In this task, we want to
detect whether a sequence of Tweets was generated by a language model or
written by a human author. Formally, our input is a Tweet sequence τ u =
{tu
1 , tu
2 , ..., tu
N }, consisting of a chronologically ordered set of N tweets from a
specific user u’s timeline. Given this input we want to learn a detector function
fθ such that, fθ (τ u) → {1, 0}; where 1 indicates that each tweet in τ u is AI-
generated and 0 means that each tweet is human written. Note that for N = 1,
this task is simply Tweet classification.
2) Human to AI Author Change Detection and Localization. In this
task, given that the input is a mixed Tweet sequence, i.e., some Tweets are AI-
generated while some are human-written, and assuming that there is only one
point in the sequence where such an author change occurs from a human to an AI
(i.e., a PLM-based text generator), we want to localize the position/tweet where
this change occurs. Formally, similar to the previous task, our input is a chrono-
logically ordered set of N Tweets from a user u’s timeline: τ u = {tu
1 , tu
2 , ..., tu
N }.
Given this timeline as input, we want to learn a function gθ such that gθ (τ u) → j;
where j ∈ [1, N ] is the index of the Tweet in the ordered set τ u where the author
change took place.
Stylometric Detection of AI-Generated Text in Twitter Timelines 5
4 Methodology
4.1 Stylometric Features
The stylometric features aim to indicate different stylistic signals from a given
piece of text. We follow a previous work on stylometry for detecting writing
style changes in literary texts [9]. In this analysis we use three categories of
features: 1) Phraseology - features which quantify how the author organizes
words and phrases when creating a piece of text (e.g., avg. word, sent. count,
etc.), 2) Punctuation - features to quantify how the author utilizes different
punctuation (e.g., avg. unique punctuation count) and 3) Linguistic Diversity
- features to quantify how the author uses different words in the writing (e.g.,
richness and readability scores). Table 1 summarizes the complete set of features
we used under each of the three stylometric feature categories.
The phraseology and punctuation feature sets are intuitive in their calcula-
tion. However, we also incorporated two complex features for linguistic diversity:
lexical richness and readability. We measure the richness of a given piece of text
by calculating the moving average type-token ratio (MTTR) metric [3]. MTTR
incorporates the average frequency of the unique words in a fixed-size moving
window (sequence of words) to measure the lexical diversity. For readability, we
use the well-established Flesch Reading Ease metric [14], that assigns a score
in-between 0-100 for the readability of a given text input.
4.2 Employing Stylometry in Human- vs. AI-Authored Tweet
detection
We follow a similar setting as in current literature where AI-generated text de-
tection is usually handled as a binary classification task where the labels are 0
and 1 (‘human written’ and ‘AI generated’) [12]. In our approach, we incorpo-
rate stylometric features as an auxiliary signal to augment the current SOTA
detectors. As shown in Fig. 2a, our proposed fusion network joins the semantic
embedding power of PLM with stylometric features.
For each set of input Tweets, we calculate normalized stylometric features
and extract the PLM-based text embedding. Let’s denote stylometric features
as sK where K is the number of features we have. From the last hidden layer
Table 1: Different stylometric feature categories and corresponding feature sets
Stylometry Analysis Features
Phraseology word count, sentence count,
paragraph count, mean and stdev of word count per sen-
tence, mean and stdev of word count per paragraph,
mean and stdev of sentence count per paragraph
Punctuation total punctuation count, mean count of special punctua-
tion (!, ’, ,, :, ;, ?, ”, -,–, @, #)
Linguistic Diversity lexical richness, readability
6 Kumarage et al.Tweet(s)
L M
Embbedding
Stylometr y
Features
Reduce
Networ k
Classification
Networ k
Fusion Networ k
+
(a) Detector model architecture: fusing
stylometric features with a PLM em-
bedding.Tweet
Timeline
Stylometr y
Analysis
Author
Changed
Feature M at
NxK
Change Point
Detection
CP M at
1xK
CP
Agreement
> t
Same
Author
Yes
No
Feature Extr action
Classification
(b) Author change detection and local-
ization: change point detection on sty-
lometry signals
Fig. 2: Proposed stylometry-based architectures
of the LM, denoted by h, we extract the vector corresponding to the CLS token
(hCLS
e ) as the text representation (e is the embedding size). After concatenating
the two vectors sK , hCLS
e , we pass them through the reduce network. Reduce
network consists of i fully-connected layers4 that learn the function rθ , reducing
the combination of sK and hCLS
e to r, where r is the reduced vector. Finally, this
reduced representation vector r is passed to the classification network (combina-
tion of j fully-connected layers followed by a softmax layer) to produce the final
classification probability, pθ (r) → pL. Here L is the label of the classification
task. The complete fusion network is trained via cross-entropy loss.
4.3 Employing Stylometry in Human-to-AI Author Change
Detection and Localization
For the task of human to AI author change detection and localization, we hy-
pothesize that when the author changes from human to AI in a Twitter timeline,
there will most likely be a significant change in the style of the text, which should
leave a segmentation in the stylometry signal. Therefore, we propose to incor-
porate change point detection on stylometric features to detect if there is an
author change. First, we briefly describe the task of change-point detection.
Change Point Detection. In time series analysis, a change point is defined
as the point in time at which an abrupt change or statistical variation occurs
within a time series. Such an abrupt change may indicate a state transition in
the system. A variety of summary statistics are used to locate the existence of
potential change points in a time series, e.g., changes in the mean, standard
deviation, or local trend slopes [27]. In this work, we used the Pruned Exact
Linear Time (PELT) algorithm [13] for detecting change points in our data. Out
4 Here i (and j) are tunable hyper-parameters, we found that i = 2 and j = 2 provided
the best results.
Stylometric Detection of AI-Generated Text in Twitter Timelines 7
of the many change point detection algorithms, we chose PELT as it performs
well and has excellent computational efficiency.
Change Point in Stylometry Signal. As shown in Fig. 2b, we first extract a
Twitter timeline’s stylometry matrix. Here, a timeline consists of N tweets, and
the number of stylometric features we use is K. This computation results in K
stylometric time series of length N . For each of these K time series, we run a
PELT analysis to determine if there exists any change points. Finally, if γ percent
of the K stylometric features agree that there is a change point, then we say that
there exists an author change within the given timeline. The percentage value
γ is a hyper-parameter and we call it the change point agreement threshold.
We define the localization point as the most agreed upon change point index
among the above mentioned γ percent of the stylometric features. If there is
no agreement between features, then a localization point is chosen at random
among those identified by the stylometric features. For simplicity, we will call
this overall methodology “stylometric change point agreement (StyloCPA)” in
the coming sections.
5 Experiments
We conducted comprehensive experiments under both tasks to explore the effec-
tiveness of the proposed models.
5.1 Datasets
In-House Dataset: The design of our research study required us to construct
sequences or “timelines” of human- and AI-authored tweets. For this, we needed
a collection of human-authored tweets and a collection of AI-generated tweets.
For a collection of human-authored tweets, we used two publicly available datasets
on specific topics (anti-vaccine5 and climate change6). We also collected tweets
on Covid-19 using the Twitter API. Note that, while of course this collection pro-
cess may potentially introduce a few AI-generated tweets into our human-tweet
collection, we do not expect these few tweets to be significant enough to skew the
results of such a large dataset. For a collection of AI-authored tweets, we gen-
erated tweets with the huggingface7 implementation of gpt2 [18], gpt2-medium,
gpt2-large and EleutherAI-gpt-neo-1.3B [7]. We fine-tuned these PLMs using
the human-authored tweets we collected so that the generated tweets matched
the style and topic of tweets in the human-authored sample. During the finetun-
ing, we kept a separate validation set to measure perplexity and perform early
stopping to ensure the model’s generation quality. As a secondary measure, we
also conducted spot-checking on generated tweets to observe whether the gener-
ated tweets were of good quality. With these tweet collections, we were able to
5 https://github.com/gmuric/avax-tweets-dataset
6 https://doi.org/10.7910/DVN/5QCCUU
7 https://huggingface.co/models
8 Kumarage et al.
construct synthetic timelines for our analysis. To build a human-authored time-
line, we queried our collection of human tweets for N tweets authored by a single
user8. This set of N tweets was then defined as a human-authored timeline. Sim-
ilarly, to build a AI-generated timeline, we sampled N of a given NLG’s tweets.
For our analysis, we varied timeline lengths (N ∈ {1, 5, 10, 20}) to understand
its affect on performance. Using the process outlined above, for each N , we con-
structed M timelines, where M = 5000/N . While the number of timelines vary
for each N , the volume of semantic information is held constant across various
N . For the change point detection and localization analysis, we needed “mixed”
timelines, i.e., a sequence of tweets where we see a human-to-AI author change.
To construct each mixed timeline N − ` human-authored tweets (on the same
topic) were sampled as above and ` AI-generated tweets (fine-tuned on the same
topic) were concatenated. For the localization results reported here we fixed
N = 25 and varied ` ∈ [1, N − 1]. We repeated this process to obtain 250 mixed
timelines. We will assist in reproducing our dataset as follows: 1) release all the
tweet-ids (or the source of the tweet-ids) used to extract the tweet content, and
2) outline the steps of how we generate the AI-generated tweets 9.
TweepFake: As a point of comparison we also applied our approach to the
public TweepFake dataset [5] which was designed for Human- v. AI-authored
tweet detection. For more information about this dataset and its construction
see [5]. Note that analysis of this dataset is comparable to the in-house dataset
with N = 1.
5.2 Experimental Settings
Since fine-tuned RoBERTa models are known to perform well for generated text
detection [5,24], we chose to use RoBERTa as the language model for our stylom-
etry fusion architecture in the task of Human- vs. AI-authored tweet detection.
RoBERTa was fine-tuned on the training dataset before extracting the embed-
dings for the proposed fusion model. We decided the number of training epochs
based on an early stopping criterion where the validation loss was calculated
on a 5% holdout set. During inference, for TweepFake and the in-house dataset
(with N = 1), the input to the model is an individual tweet. However, for the
cases where timeline length N > 1, we append all the tweets in a timeline with
newline separators to create a single input text.
For the task of human to AI author change detection and localization, we
used the StyloCPA model described in the Methodology section. In order to
select the agreement threshold γ, we performed grid search over a range of γ
values and found that γ = 0.15 resulted in the best overall localization accuracy.
8 If N tweets were not available from a single user we instead collected tweets from
multiple users. Note, that a single user’s authorship is not a requirement for our
analysis but helps with consistent style when possible.
9 The data generation code is available at https://github.com/stresearch/machine-
gen-twitter.git
Stylometric Detection of AI-Generated Text in Twitter Timelines 9
5.3 Baselines for Comparison
For the task of human- vs. AI-authored tweet detection, we use the following two
categories of baselines.
Naive Feature-based: For a naive baseline, we combine a feature extrac-
tion method with a classifier, without performing any finetuning. For the fea-
ture extraction we used bag-of-words (BOW), word2vec (w2v) [17], BERT and
RoBERTa. We then used a classifier on top of the extracted features. While we
experimented with xgboost, logistic regression, and random forest classifiers, for
brevity, we only report the results associated with xgboost in Table 2 as this
was the top performer.
Fine-tuned LM based: Here, we follow previous works [5, 24] in AI generated
text detection and use LM-based classifiers, viz., BERT and RoBERTa (fine-
tuned on the training data) as the baselines.
Similarly for the human to AI author change detection and localization task, we
use the following two categories of baselines:
Fine-tuned LM based: For this task, for a given timeline, we calculated the
classification probability for each tweet i.e, the probability that a tweet is gener-
ated by a human or AI using the top performing LM-based classifiers from the
previous task. We then used the change point detection algorithm discussed in
the Methodology section to detect if there is a change in the detection probability
time series.
Style change classifiers: We used the top two models from the PAN “style
change detection” task [28]; 1) PAN BERT [11]: BERT model on stylometry
signals, 2) PAN Stack En [25]: stacked ensemble model based on BERT embed-
ding. These PAN style change detection models identify author changes within a
multi-author document. They assume that an author change only occurs at the
paragraph level. In the analysis reported here, when working with the PAN mod-
els we first converted each tweet timeline into a document where each paragraph
is a single tweet from the timeline.
5.4 Experimental Results
Table 2 summarizes the results of the task Human- vs. AI-authored tweet detec-
tion. It is evident that the stylometric fusion seems to augment the performance
of detecting AI-generated tweets by a significant margin. By looking at the naive
classifiers, standalone stylometric features are a good signal in discriminating AI-
generated tweets. In particular, stylometric features outperform BOW and w2v
embeddings. However, stylometric features are not as powerful as the pre-trained
LM embeddings for this task, which is intuitive given the high volume of seman-
tic information retained by these PLMs at the pre-training stage. The results
on the TweepFake dataset further confirm the claims mentioned above. As seen
in the rightmost column of Table 2, our stylometry LM fusion model outper-
forms the current SOTA RoBERTa baseline on TweepFake. Overall, the models
tend to perform better on TweepFake compared to our in-house dataset. One
10 Kumarage et al.
possible explanation for this difference would be that our dataset was generated
purely with SOTA NLG models, in contrast, TweepFake used a mix of primitive
and SOTA generators. This may result in our AI-generated tweets being more
realistic and thus harder to detect.
Table 2: Proposed stylometry fusion model performance (accuracy) on Human-
vs. AI-Authored Tweet detection.
Dataset → In-House TweepFake
Model ↓ N = 1 N = 5 N = 10 N = 20
XGB BOW 0.718 0.819 0.879 0.951 0.792
XGB W2V 0.732 0.873 0.911 0.963 0.845
XGB Stylo (ours) 0.771 0.891 0.909 0.958 0.847
XGB BERT EMB 0.796 0.902 0.911 0.972 0.853
XGB RoBERTa EMB 0.798 0.910 0.913 0.974 0.857
BERT FT 0.802 0.913 0.919 0.979 0.891
RoBERTa FT 0.807 0.919 0.927 0.981 0.896
RoBERTa FT Stylo (ours) 0.875 0.942 0.961 0.992 0.911
In Table 2 we also present how the models perform on different timeline
lengths (N ). When the number of tweets in the timeline decreases, the semantic
information that resides in a given timeline also decreases, therefore, making it
difficult for the classifiers to discriminate AI tweets from human tweets.50 150 250
# Timelines in Training Data
78
80
82
84
86
88
90
92
94
96
Avg. Accuracy
StyloCPA
RoBERTa
RoBERTa_Stylo
PAN_Stack_En
PAN_BERT
Fig. 3: Accuracy in detecting mixed
timelines as a function of training set
size.
However, when the timeline is small,
we see an accuracy gain from stylo-
metric fusion. This may suggest that
stylometric signals help compensate
for performance loss due to low se-
mantic information.
Fig. 3 shows our results on the hu-
man to AI author change detection
task as a function of different train-
ing set sizes. We see that the pro-
posed StyloCPA model performs well
compared to most baselines. In fact,
when the number of training samples
is small (50), it has the best perfor-
mance across all the models. This is
rather impressive because unlike fine-tuned PLM-based detectors, StyloCPA has
few learnable parameters and performs well with limited training samples.
Table 3 shows the localization (i.e., detecting the time of an author change)
results for different window sizes (i.e, true positive occurs when the predicted
change point is within a window of ±W points from the actual change point).
Our StyloCPA has the best performance compared to PLM-based detectors. In
Stylometric Detection of AI-Generated Text in Twitter Timelines 11
the PLM-based detectors, the error in detecting AI vs. human tweets is prop-
agated into the localization task. Therefore, it cannot precisely pinpoint the
change; yet it would be in a close region. We see this by observing the increase
in accuracies when the window size W increases. However, in StyloCPA, pin-
pointing the author change is more feasible and accurate, given that it detects
an author change based on an agreement between all the stylometric features.
Table 3: Performance on detecting the change-point.
Model W = 0 W = 1 W = 2
StyloCPA 0.822 0.871 0.892
RoBERTa 0.745 0.824 0.853
RoBERTa Stylo 0.795 0.865 0.889
PAN Stack En 0.672 0.752 0.794
PAN BERT 0.761 0.843 0.862
5.5 Further Analysis
Here we further study how different variations in data, generators, and stylome-
try features affect the proposed models. Please note that we use the label T1 for
human- vs. AI-authored tweet detection and T2 for human-to-AI author change
detection and localization in the below section.
Does Topic Change Affect Performance? As seen in Figure 4a, we do not
see a significant change in results when the topic changes. All the stylometry
features incorporated in our method are intended to identify style changes in
tweets. Though the topic of a tweet changes, the writing style attributes of a
given author is relatively invariant. Consequently, we would not see a significant
difference in the performance.
Are Bigger Generators Harder to Detect? Figure 4b shows the perfor-
mance of the stylometry-LM fused detector across multiple generators. As ex-
pected, we see a slight decrease in performance when the size of the generator
increases (large generators capable of mimicking human writing style well).
Which Stylometry Features Were The Most Important? Figure 4c shows
each stylometry category’s aggregated average importance score for T1 and T2
across all the timeline sizes. As we see, punctuation and phraseology features
are the most important in contrast to the linguistic features. This maybe be
because the linguistic features require long text sequences to present a more
accurate score. Therefore, the readability and diversity scores we extract might
not be a good signal for detecting AI-generated tweets. This remark is further
evident by the increased lexical feature importance from T1 to T2. T2 has larger
timelines (N = 25) compared to the T1 timelines (N ∈ {1, 10, 20}).
12 Kumarage et al.Climate Covid-19 Anti-Vaccine
Topic
88
90
92
94
96
98
Avg. Accuracy
T1
T2
(a)100 200 300 400 500 600 700 800 900 1000
Generator Size (Millions)
93
94
95
96
97
98
99
Avg. Accuracy
T1
T2 (b)T1 T2
Tasks
0
10
20
30
40
50
60
Feature Importance
Linguistic
Punctuation
Phraseology (c)
Fig. 4: Further analysis on different variations in data and features; (a) Perfor-
mance on different topics, (b) Performance vs. generator size (number of param-
eters), and (c) Features importance in each stylometry feature category.
6 Conclusion
In this paper, we studied the novel application of incorporating stylometry to
quantify stylistic changes in AI-generated text in Twitter timelines. We pro-
posed two simple architectures to utilize three categories of stylometric features
towards 1) discriminating between human-written and AI-generated tweets and
2) detecting if and when an AI starts to generate tweets in a given Twitter
timeline. We created an in-house dataset to emulate these two tasks. A com-
prehensive set of experiments on the in-house data and an existing benchmark
dataset shows that the proposed architectures perform well in augmenting the
current PLM-based AI-generated text detector capabilities. For future work, it
would be interesting to see how capable stylistic signals are towards attributing
a given AI tweet to a corresponding generator.
7 Acknowledgement
This research was developed with funding from the Defense Advanced Research
Projects Agency (DARPA). The views, opinions and/or findings expressed are
those of the author and should not be interpreted as representing the official
views or policies of the Department of Defense or the U.S. Government.
References
1. Bakhtin, A., et al.: Real or fake? learning to discriminate machine from human
generated text. arXiv preprint arXiv:1906.03351 (2019)
2. Bhargava, M., et al.: Stylometric analysis for authorship attribution on twitter. In:
ICBDA. pp. 37–47. Springer (2013)
3. Covington, M.A., McFall, J.D.: Cutting the gordian knot: The moving-average
type–token ratio (mattr). Journal of quantitative linguistics 17(2), 94–100 (2010)
4. Efthimion, P.G., et al.: Supervised machine learning bot detection techniques to
identify social twitter bots. SMU Data Science Review 1(2), 5 (2018)
5. Fagni, T., et al.: Tweepfake: About detecting deepfake tweets. Plos one 16(5),
e0251415 (2021)
Stylometric Detection of AI-Generated Text in Twitter Timelines 13
6. Feng, S., et al.: Heterogeneity-aware twitter bot detection with relational graph
transformers. In: AAAI. vol. 36, pp. 3977–3985 (2022)
7. Gao, L., et al.: The pile: An 800gb dataset of diverse text for language modeling.
arXiv preprint arXiv:2101.00027 (2020)
8. Gehrmann, S., et al.: Gltr: Statistical detection and visualization of generated text.
In: ACL: System Demonstrations. pp. 111–116 (2019)
9. G´omez-Adorno, H., et al.: Stylometry-based approach for detecting writing style
changes in literary texts. Computaci´on y Sistemas 22(1), 47–53 (2018)
10. Ippolito, D., et al.: Automatic detection of generated text is easiest when humans
are fooled. arXiv preprint arXiv:1911.00650 (2019)
11. Iyer, A., Vosoughi, S.: Style change detection using bert. (2020)
12. Jawahar, G., et al.: Automatic detection of machine generated text: A critical
survey. In: COLING. pp. 2296–2309 (2020)
13. Killick, R., et al.: Optimal detection of changepoints with a linear computational
cost. Journal of the American Statistical Association 107(500), 1590–1598 (2012)
14. Kincaid, J.P., et al.: Derivation of new readability formulas (automated readability
index, fog count and flesch reading ease formula) for navy enlisted personnel. Tech.
rep., Naval Technical Training Command Millington TN Research Branch (1975)
15. Knauth, J.: Language-agnostic twitter-bot detection. In: RANLP. pp. 550–558
(2019)
16. Kushnareva, L., et al.: Artificial text detection via examining the topology of at-
tention maps. In: EMNLP. pp. 635–649 (2021)
17. Mikolov, T., et al.: Efficient estimation of word representations in vector space.
arXiv preprint arXiv:1301.3781 (2013)
18. Radford, A., et al.: Language models are unsupervised multitask learners. OpenAI
blog 1(8), 9 (2019)
19. Schuster, T., et al.: The limitations of stylometry for detecting machine-generated
fake news. Computational Linguistics 46(2), 499–510 (2020)
20. Shu, K., et al.: Deep headline generation for clickbait detection. In: ICDM. pp.
467–476. IEEE (2018)
21. Shu, K., et al.: Fact-enhanced synthetic news generation. In: AAAI. vol. 35, pp.
13825–13833 (2021)
22. Shukla, H., et al.: Enhanced twitter bot detection using ensemble machine learning.
In: ICICT. pp. 930–936. IEEE (2021)
23. Solaiman, I., et al.: Release strategies and the social impacts of language models.
arXiv preprint arXiv:1908.09203 (2019)
24. Stiff, H., Johansson, F.: Detecting computer-generated disinformation. Interna-
tional Journal of Data Science and Analytics 13(4), 363–383 (2022)
25. Strøm, E.: Multi-label style change detection by solving a binary classification
problem. In: CLEF (Working Notes). pp. 2146–2157 (2021)
26. Tesfagergish, S.G., et al.: Deep fake recognition in tweets using text augmentation,
word embeddings and deep learning. In: ICCSA. pp. 523–538. Springer (2021)
27. Truong, C., et al.: Selective review of offline change point detection methods. Signal
Processing 167, 107299 (2020)
28. Zangerle, E., et al.: Overview of the Style Change Detection Task at PAN 2021.
In: Faggioli, G., Ferro, N., Joly, A., Maistro, M., Piroi, F. (eds.) CLEF 2021 Labs
and Workshops, Notebook Papers. CEUR-WS.org (2021)
29. Zellers, R., et al.: Defending against neural fake news. NeurIPS 32 (2019)
30. Zhong, W., et al.: Neural deepfake detection with factual structure of text. In:
EMNLP. pp. 2461–2470 (2020)