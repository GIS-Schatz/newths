from django.shortcuts import render

# Create your views here.
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 우리가 예측한 평점과 실제 평점간의 차이를 MSE로 계산
def get_mse(pred, actual):
    # 평점이 있는 실제 맥주이름 추출
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# 특정 맥주와 비슷한 유사도를 가지는 맥주 Top_N에 대해서만 적용 -> 시간오래걸림
def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
    # 사용자-아이템 평점 행렬 크기만큼 0으로 채운 예측 행렬 초기화
    pred = np.zeros(ratings_arr.shape)

    # 사용자-아이템 평점 행렬의 맥주 개수만큼 루프
    for col in range(ratings_arr.shape[1]):
        # 유사도 행렬에서 유사도가 큰 순으로 n개의 데이터 행렬의 인덱스 반환
        top_n_items = [np.argsort(item_sim_arr[:, col])[:-n - 1:-1]]
        # 개인화된 예측 평점 계산 : 각 col 맥주별(1개), 2496 사용자들의 예측평점
        for row in range(ratings_arr.shape[0]):
            pred[row, col] = item_sim_arr[col, :][top_n_items].dot(
                ratings_arr[row, :][top_n_items].T)
            pred[row, col] /= np.sum(item_sim_arr[col, :][top_n_items])

    return pred


# 사용자가 먹어본 맥주 제외
def user_not_tried_beer(user_item_matrix, userid):
    # ID로 입력 받은 사용자의 맥주 정보를 추출해 Series로 변환
    # 반환된 user_rating는 beer('맥주이름')을 인덱스로 가지는 Series객체?
    user_rating = user_item_matrix.loc[userid, :]

    # user_rating이 0보다 크면 기존에 마신 맥주
    # 0보다큰 인덱스를 추출해 list객체로 만듦
    tried = user_rating[user_rating > 0].index.tolist()

    # 모든 맥주명을 list객체로
    beer_list = user_item_matrix.columns.tolist()

    # tried에 해당하는 맥주는 beer_list에서 제외
    not_tried = [beer for beer in beer_list if beer not in tried]

    return not_tried


#예측 평점 DataFrame에서 ID인덱스와 not_tried로 들어온 맥주명 추출 후
#가장 예측 평점이 높은 순으로 정렬

def recommand_beer_by_id(pred_df, userid, not_tried, top_n):
    recommand_beer = pred_df.loc[userid, not_tried].sort_values(ascending=False)[:top_n]
    return recommand_beer


# 평점, Aroma, Flavor, Mouthfeel 중 피처 선택 후 유사도 계산
def recomm_feature(df, col):
    feature = col
    ratings = df[['평점', 'ID', '맥주이름']]

    # 피벗 테이블을 이용해 유저-아이디 매트릭스 구성
    user_item_matrix = ratings.pivot_table('평점', index='ID', columns='맥주이름')

    # nan -> 0으로 채움
    user_item_matrix = user_item_matrix.fillna(0)

    # 유사도 계산을 위해 트랜스포즈
    user_item_matrix_T = user_item_matrix.T

    # 아이템 - 유저 매트릭스로부터 코사인 유사도 구하기
    item_user_cos_sim = cosine_similarity(user_item_matrix_T, user_item_matrix_T)

    # cosine_similarity()로 반환된 넘파이 행렬에 맥주이름을 넣어서 (인덱스와 컬럼에) DataFrame로 변환
    item_user_cos_sim_df = pd.DataFrame(data=item_user_cos_sim,
                                        index=user_item_matrix.columns,
                                        columns=user_item_matrix.columns)

    return item_user_cos_sim_df


# 해당 맥주와 유사한 유사도 5개 추천
def recomm_beer(item_user_cos_sim_df, beer_name):
    # 해당 맥주와 유사도가 높은 맥주 5개만 추천
    return item_user_cos_sim_df[beer_name].sort_values(ascending=False)[1:4]


def index(request):
    return render(request, 'newths/index.html')


def ver1(request):
    beer_list = pd.read_csv('data/맥주이름.csv', encoding='utf-8', index_col=0)
    beer_year = pd.read_csv('data/맥주_연도별평점.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('data/정제된데이터.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('data/대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('data/전체맥주클러스터링.csv', encoding='utf-8', index_col=0)
    beer_data = pd.read_csv('data/맥주정보데이터.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['맥주이름']
    cluster_3 = cluster_3.values

    if request.method == 'POST':
        beer_name = request.POST.get('beer', '')
        detail = request.POST.get('detail', '')
        df_aroma = recomm_feature(ratings, 'Aroma')
        df_flavor = recomm_feature(ratings, 'Flavor')
        df_mouthfeel = recomm_feature(ratings, 'Mouthfeel')

        if detail == 'Aroma':
            df = df_aroma * 0.8 + df_flavor * 0.1 + df_mouthfeel * 0.1
        if detail == 'Flavor':
            df = df_aroma * 0.1 + df_flavor * 0.8 + df_mouthfeel * 0.1
        if detail == 'Mouthfeel':
            df = df_aroma * 0.1 + df_flavor * 0.1 + df_mouthfeel * 0.8

        result = recomm_beer(df, beer_name)
        result = result.index.tolist()

        # 클러스터링 결과
        tmp_cluster = []
        category = []
        food = []
        for i in range(3):
            target = cluster_all[cluster_all['맥주이름'] == result[i]]
            target = target[['Aroma', 'Appearance', 'Flavor', 'Mouthfeel', 'Overall']]
            target = target.values[0]
            tmp_cluster.append(target)

            try:
                category.append(beer_data[beer_data['맥주이름'] == result[i]]['Main Category'].values[0])
                food.append(beer_data[beer_data['맥주이름'] == result[i]]['Paring Food'].values[0])
            except:
                category.append('수집되지 않았습니다.')
                food.append('수집되지 않았습니다.')


        # 연도별 평점 결과
        tmp_year = []
        tmp_ratings = []
        for i in range(3):
            target = beer_year[beer_year['맥주이름'] == result[i]]
            target_year = target['년'].tolist()
            target_rating = target['평점'].tolist()
            tmp_year.append(target_year)
            tmp_ratings.append(target_rating)

        # 넘겨줄 데이터 Json 변환
        targetdict = {
            'beer_name': result,
            'beer_cluster1': tmp_cluster[0].tolist(),
            'beer_cluster2': tmp_cluster[1].tolist(),
            'beer_cluster3': tmp_cluster[2].tolist(),
            'cluster1': cluster_3[0].tolist(),
            'cluster2': cluster_3[1].tolist(),
            'cluster3': cluster_3[2].tolist(),
            'tmp_year': tmp_year,
            'tmp_ratings': tmp_ratings
        }

        targetJson = json.dumps(targetdict)

        return render(request, 'newths/ver1_result.html',
                      {'result': result, 'beer_list': beer_list, 'targetJson': targetJson,
                       'category': category, 'food': food})
    else:
        return render(request, 'newths/ver1.html', {'beer_list': beer_list})


def ver2(request):
    beer_list = pd.read_csv('data/맥주이름.csv', encoding='utf-8', index_col=0)
    beer_year = pd.read_csv('data/맥주_연도별평점.csv', encoding='utf-8', index_col=0)
    ratings = pd.read_csv('data/정제된데이터.csv', encoding='utf-8', index_col=0)
    cluster_3 = pd.read_csv('data/대표군집클러스터링.csv', encoding='utf-8', index_col=0)
    cluster_all = pd.read_csv('data/전체맥주클러스터링.csv', encoding='utf-8', index_col=0)
    beer_data = pd.read_csv('data/맥주정보데이터.csv', encoding='utf-8', index_col=0)
    beer_list = beer_list['맥주이름']
    cluster_3 = cluster_3.values

    if request.method == 'POST':
        name = request.POST.get('name', '')
        beer = []
        rating = []
        for i in range(1,6):
            beer.append(request.POST.get('beer'+str(i), ''))
            rating.append((request.POST.get('rating'+str(i), '')))

        for i in range(len(beer)):
            tmp = []
            tmp.append(name)
            tmp.append(beer[i])
            tmp.append(float(rating[i]))
            tmp = pd.DataFrame(data=[tmp], columns=['ID','맥주이름','평점'])
            ratings = pd.concat([ratings, tmp])

        username = name
        # 피벗 테이블을 이용해 유저-아이디 매트릭스 구성
        user_item_matrix = ratings.pivot_table('평점', index='ID', columns='맥주이름')
        # nan -> 0으로 채움
        user_item_matrix = user_item_matrix.fillna(0)
        # user_item_matrix -> Collaborating Filtering(협업필터링) -> item_user_matrix로 변환
        user_item_matrix_T = user_item_matrix.T

        # 아이템 - 유저 매트릭스로부터 코사인 유사도 구하기
        item_user_cos_sim = cosine_similarity(user_item_matrix_T, user_item_matrix_T)
        # cosine_similarity()로 반환된 넘파이 행렬에 맥주이름을 넣어서 (인덱스와 컬럼에) DataFrame로 변환
        item_user_cos_sim_df = pd.DataFrame(data=item_user_cos_sim,
                                            index=user_item_matrix.columns,
                                            columns=user_item_matrix.columns)

        # top_n과 비슷한 맥주만 추천에 사용
        ratings_pred = predict_rating_topsim(user_item_matrix.values, item_user_cos_sim_df.values, n=5)   # 계산된 예측 평점 데이터는 DataFrame으로 재생성
        # 계산된 예측 평점 데이터는 DataFrame으로 재생성
        ratings_pred_matrix = pd.DataFrame(data=ratings_pred,
                                           index=user_item_matrix.index,
                                           columns=user_item_matrix.columns)
        # 유저가 먹지 않은 맥주이름 추출
        not_tried = user_not_tried_beer(user_item_matrix, username)
        # 아이템 기반의 최근접 이웃 CF로 맥주 추천
        recommand_beer = recommand_beer_by_id(ratings_pred_matrix, username, not_tried, top_n=3)
        recommand_beer = pd.DataFrame(data=recommand_beer.values, index=recommand_beer.index,
                                      columns=['예측평점'])
        # 추천 결과로 나온 맥주이름만 추출
        result = recommand_beer.index.tolist()

        # 클러스터링 결과
        tmp_cluster = []
        category = []
        food = []
        for i in range(3):
            target = cluster_all[cluster_all['맥주이름'] == result[i]]
            target = target[['Aroma', 'Appearance', 'Flavor', 'Mouthfeel', 'Overall']]
            target = target.values[0]
            tmp_cluster.append(target)

            try :
                category.append(beer_data[beer_data['맥주이름']==result[i]]['Main Category'].values[0])
                food.append(beer_data[beer_data['맥주이름']==result[i]]['Paring Food'].values[0])
            except :
                category.append('수집되지 않았습니다.')
                food.append('수집되지 않았습니다.')

        tmp_year = []
        tmp_ratings = []
        for i in range(3):
            target = beer_year[beer_year['맥주이름'] == result[i]]
            target_year = target['년'].tolist()
            target_rating = target['평점'].tolist()
            tmp_year.append(target_year)
            tmp_ratings.append(target_rating)

        targetdict = {
            'beer_name': result,
            'beer_cluster1': tmp_cluster[0].tolist(),
            'beer_cluster2': tmp_cluster[1].tolist(),
            'beer_cluster3': tmp_cluster[2].tolist(),
            'cluster1': cluster_3[0].tolist(),
            'cluster2': cluster_3[1].tolist(),
            'cluster3': cluster_3[2].tolist(),
            'tmp_year': tmp_year,
            'tmp_ratings': tmp_ratings
        }

        targetJson = json.dumps(targetdict)

        return render(request, 'newths/ver2_result.html',
                      {'result': result, 'beer_list': beer_list,'targetJson': targetJson,
                       'category':category, 'food':food})

    else:
        return render(request, 'newths/ver2.html', {'beer_list': beer_list})