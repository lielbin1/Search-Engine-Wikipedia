# Search Enging Wikipedia - IR project
Search engine for entire wikipedia corpus made as minor project in course taken in semester 3, Information Retrieval  course.

## **Utility**
- parses wikipedia dump and makes inverted index
- merges index files and split them into smaller chunks
- main query program that returns results in less than 5 seconds

## **Data**

● Entire Wikipedia dump in a shared Google Storage bucket. [Download](https://drive.google.com/file/d/1QMpM1CSn6j8Hwu5AabTqTQ1km9xCzSEV/view)

● Pageviews for articles.

● Queries and a ranked list of up to 100 relevant results for them, split into train (30
queries+results given to you in queries_train.json) and test (held out for evaluation).

## **Code**

● search_frontend.py: Flask app for search engine frontend.

● run_frontend_in_colab.ipynb: notebook showing how to run your search engine's frontend
in Colab for development purposes.

● run_frontend_in_gcp.sh: Deploying the search engine in GCP.

● startup_script_gcp.sh: a shell script that sets up the Compute Engine

### Constructing the Inverted Index
- XML parsing
- Data preprocessing
- Tokenization
- Case folding
- remove stop words
- Stemming
- Posting List / Inverted Index Creation
- Optimize

## **Features**
Support for Field Queries. 
Fields include Title, Infobox, Body, Category, Links, and References of a Wikipedia page. 

Index size should be less than 1⁄4 of dump size.

Scalable index construction

Search Functionality

Index creation time: less than 150 secs.

Inverted index size: 1/4th of entire wikipedia corpus
