import sys

from collections import defaultdict

import mariadb
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from .application import DataSource

dataSource = DataSource()

class TFIDF_Model:
    def __init__(self):
        self.username = dataSource.username
        self.password = dataSource.password
        self.host = dataSource.host
        self.port = dataSource.port
        self.database = dataSource.database
        
    # MariaDB Connection
    def getConnection(self):
        try:
            connection = mariadb.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
        except mariadb.Error as e:
            print(f"Error : Connecting to MariaDB !\n{e}")
            sys.exit(1)
        return connection.cursor()
    
    # Get project id from some table by using member id
    def getProjectIdFromTable(self, table, memberId):
        conn = self.getConnection()

        query = f"SELECT project_id FROM {table} WHERE member_id={memberId}"
        conn.execute(query)
        output = [e[0] for e in conn.fetchall()]
        conn.close()
        return output
    
    # Get Corpus using some column
    def getCorpus(self, column):
        conn = self.getConnection()

        if column == "tag":
            query = "SELECT project_id, REPLACE(name, ' ', '') FROM project_tag as P INNER JOIN tags as T ON P.tags_id = T.id WHERE P.project_id in (SELECT project_id FROM admin_approvals WHERE approval=1)"

            conn.execute(query)
            df_tag = pd.DataFrame(conn.fetchall())
            df_tag.columns = ["projectId", "tagName"]
            corpus = {}

            for projectId in range(df_tag.projectId.min(), df_tag.projectId.max()+1):
                if projectId in df_tag.projectId:
                    tag_list = df_tag.loc[df_tag.projectId==projectId]["tagName"].values
                    tag_string = " ".join(tag_list)
                    corpus[projectId] = tag_string
            corpus = {key:value for key, value in corpus.items() if value != ''}
        else:
            query = f'SELECT id, {column} FROM projects WHERE id in (SELECT project_id FROM admin_approvals WHERE approval=1)'

            conn.execute(query)
            corpus = {id:value for id, value in conn.fetchall()}
        
        conn.close()
        return corpus
    
    # Calculate tf-idf value from corpus
    def corpus2tfidf(self, corpus):
        tfidfv = TfidfVectorizer().fit(corpus.values())
        return tfidfv.transform(corpus.values()).toarray()

    # Calcualte cosine similarity from tf-idf
    def tfidf2similarity(self, corpus, tfidf_matrix):
        cosine_similarity = np.dot(tfidf_matrix, tfidf_matrix.T)
        cosine_similarity /= np.linalg.norm(tfidf_matrix, axis=1)**2

        df_matrix = pd.DataFrame(cosine_similarity)
        df_matrix.columns = corpus.keys()
        df_matrix.index = corpus.keys()
        return df_matrix
    
    # Get recommendation from project id
    def getRecommendationFromProjectId(self, projectId):
        project_representation = np.concat([self.corpus2tfidf(self.getCorpus(column)) 
            for column in ["title", "description", "description_detail", "tag"]], axis=1)

        result = self.tfidf2similarity(corpus=self.getCorpus("title"), tfidf_matrix=project_representation)
        return result.loc[projectId].sort_values(ascending=False).index.values
    
    # Aggregation recommendation rakings
    def aggregationIndices(self, rankings):
        scoreboard = defaultdict(int)
        for ranking in rankings:
            num_items = len(ranking)
            for rank, item in enumerate(ranking):
                scoreboard[item] += num_items - rank
        return [int(e[0]) for e in sorted(scoreboard.items(), key=lambda x: x[1], reverse=True)]

    # Get Recommendation from member id
    def getRecommendationFromMemberId(self, memberId):
        member_list = self.getProjectIdFromTable("liked_projects", memberId)
        if not member_list:
            return []
        rankings = [self.getRecommendationFromProjectId(id) for id in member_list]
        recommendation = self.aggregationIndices(rankings)
        recommendation = [r for r in recommendation if r not in member_list]
        return recommendation
