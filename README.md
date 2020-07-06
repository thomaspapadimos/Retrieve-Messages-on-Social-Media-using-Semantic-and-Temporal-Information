# Event-based Messages Retrieval on Social Media Using Semantic and Temporal Information



## Abstract

Ιn social media platforms, such as Twitter, the content that is produced is often influenced by current ongoing events, as people tend to share opinions and relevant information about them. Given the limited length of the messages in platforms such as Twitter, messages related to different events can have exhibit high textual similarity due to a significant overlap in common terms. At the same time, messages from the same event can be highly dissimilar in terms of semantic proximity, making it difficult to associate them with a specific event-related query and subsequently retrieve them. As events are characterized by the time taking place, publication time of messages becomes an important feature to estimate relevance.
The purpose of this project is to tackle the problem of event-based information retrieval in social media platforms by using methods that are suitable for the retrieval of short messages. These methods consider the semantic content and the temporal characteristics of both messages and event-related queries. Given a query we first retrieve an initial set of potentially relevant documents, which then are reranked by the neural network approach proposed in this work. More precisely, Convolutional Neural Networks (CNNs) that have recently shown very good results in natural language processing (NLP) tasks and Information Retrieval (Neural IR), have been used to combine the semantic features (word embeddings) of both queries and documents. Different statistical techniques were also used to model the temporal characteristics of the initially retrieved messages with respect to the query. These temporal features are fed into the neural network as additional features to the textual ones, and the model is trained to re-rank the documents.

![Architect](https://github.com/thomaspapadimos/Retreival-messages/blob/master/ARCHITECT.png)

About author
------------
- Thomas Papadimos
- thomas5papad@gmail.com
- 2018-2019