# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 19:54:07 2019

@author: Chacrew
"""

from cdqa.retriever import BM25Retriever
import pandas as pd
df = pd.read_csv('data/bnpp_newsroom_v1.1/jungchat_result_191014.csv')
retriever = BM25Retriever(ngram_range=(1, 2), max_df=1.00,min_df=1, stop_words=None)
retriever.fit(df)
best_idx_scores = retriever.predict(query='희망두배 청년통장이 뭐에요?')
print(best_idx_scores)
