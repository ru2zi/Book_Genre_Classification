# Book_Genre_Classification

['sentence-transformers/all-mpnet-base-v2'] 굿 <br>
['joeddav/distilbert-base-uncased-agnews-student'] <br>
['andi611/distilbert-base-uncased-ner-agnews'] <br>
['sileod/deberta-v3-base-tasksource-nli'] <br>
['microsoft/deberta-v3-base'] <br>
['valurank/finetuned-distilbert-news-article-categorization']굿 <br>
['facebook/bart-base'] 굿 <br>


## Total model
model_names = [
    'abhishek/autonlp-bbc-news-classification-37229289',
    'distilbert-base-uncased',
    'valurank/finetuned-distilbert-news-article-categorization',
    'mosesju/distilbert-base-uncased-finetuned-news',
    'fabriceyhc/bert-base-uncased-ag_news',
    'philschmid/distilbert-base-multilingual-cased-sentiment',
    'arjuntheprogrammer/distilbert-base-multilingual-cased-sentiment-2',
    'microsoft/deberta-v3-base',
    'sileod/deberta-v3-base-tasksource-nli',
    'mrm8488/bert-mini-finetuned-age_news-classification',
    'JiaqiLee/bert-agnews',
    'andi611/distilbert-base-uncased-ner-agnews',
    'lucasresck/bert-base-cased-ag-news',
    'nateraw/bert-base-uncased-ag-news',
    'joeddav/distilbert-base-uncased-agnews-student',
    'bertugmirasyedi/deberta-v3-base-book-classification',
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/all-MiniLM-L6-v2',
    'cross-encoder/nli-deberta-base',
    'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    'typeform/distilbert-base-uncased-mnli',
    'facebook/bart-base',
    'decapoda-research/llama-7b-hf',
    'fxmarty/tiny-llama-fast-tokenizer',
    'decapoda-research/llama-13b-hf',
    'huggyllama/llama-7b',
    'decapoda-research/llama-65b-hf',
    'facebook/opt-66b',
    'KoboldAI/OPT-6.7B-Nerybus-Mix',
    'facebook/opt-1.3b',
    'facebook/opt-125m',
    'bigscience/bloom-560m',
    'bigscience/test-bloomd-6b3',
    'bigscience/bloom-7b1',
    'xlm-roberta-base',
    'roberta-base',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'openai-gpt'
]

## 조언

1. (전처리) Pre-trained model을 사용하려면 그 모델이 어떤 데이터로 학습했는지를 파악하고 그러한 데이터의 형식에 맞추는 것이 중요함
   그래서 Pre-trained model 들은 자체적인 tokenizer를 활용해서 전처리를 할수 있게 함수가 이미 마련되어 있어서 그걸 사용하면 됨
   -요약- 우리가 따로 전처리할 건 없고 그냥 Pre-trained모델의 전처리 함수에 믿고 맡겨라
2. (Multimodal) 이미지 데이터를 보니 특징을 파악하기 힘들어서 이미지 관련 모델은 안쓰는게 좋을 듯
   -요약- Text classification에 더 집중해라
3. (하이퍼 파라미터) 여러모델들을 바꿔보면서 학습해서 좋은 모델을 찾아야하겠지만 하나하나 하이퍼 파라미터를 최적화하기는 힘드니깐 일단 사이트에서 추천하는 default값을 기본으로 모델을 돌려보
   서 검증해보고 좋은 것만 최적화해봐라라저기에 나와있지 않은 파라미터들은 이미 pre-trained모델에서 학습되어 있는거니깐 건드릴 생각하지말것
