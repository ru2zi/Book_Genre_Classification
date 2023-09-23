# Book_Genre_Classification

['sentence-transformers/all-mpnet-base-v2'] 굿
['joeddav/distilbert-base-uncased-agnews-student']
['andi611/distilbert-base-uncased-ner-agnews']
['sileod/deberta-v3-base-tasksource-nli']
['microsoft/deberta-v3-base']
['valurank/finetuned-distilbert-news-article-categorization']굿
['facebook/bart-base'] 굿

## 조언

1. (전처리) Pre-trained model을 사용하려면 그 모델이 어떤 데이터로 학습했는지를 파악하고 그러한 데이터의 형식에 맞추는 것이 중요함
   그래서 Pre-trained model 들은 자체적인 tokenizer를 활용해서 전처리를 할수 있게 함수가 이미 마련되어 있어서 그걸 사용하면 됨
   -요약- 우리가 따로 전처리할 건 없고 그냥 Pre-trained모델의 전처리 함수에 믿고 맡겨라
2. (Multimodal) 이미지 데이터를 보니 특징을 파악하기 힘들어서 이미지 관련 모델은 안쓰는게 좋을 듯
   -요약- Text classification에 더 집중해라
3. (하이퍼 파라미터) 여러모델들을 바꿔보면서 학습해서 좋은 모델을 찾아야하겠지만 하나하나 하이퍼 파라미터를 최적화하기는 힘드니깐 일단 사이트에서 추천하는 default값을 기본으로 모델을 돌려보
   서 검증해보고 좋은 것만 최적화해봐라라저기에 나와있지 않은 파라미터들은 이미 pre-trained모델에서 학습되어 있는거니깐 건드릴 생각하지말것
