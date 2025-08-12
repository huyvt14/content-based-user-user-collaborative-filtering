# đang dùng mới nhất
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, jsonify

app = Flask(__name__)
file_path = '/data.xlsx'
file_path1 = '/rating.xlsx'
df_ratings = None
df_item = None
cosine_sim = None
user_similarity_df = None
rating_matrix = None

def ini_cosine_sim():
    global df_item

    bins = [0, 300000, 700000, 1000000, 2000000, 3000000]
    labels = ['0-300', '300-700', '700-1000', '1000-2000', '2000-3000']
    df_item['price_range'] = pd.cut(df_item['current_price'], bins=bins, labels=labels)

    encoder = OneHotEncoder()
    encoded_attributes = encoder.fit_transform(df_item[['category', 'subcategory', 'price_range']])
    encoded_df = pd.DataFrame(encoded_attributes.toarray(), 
                            columns=encoder.get_feature_names_out(['category', 'subcategory', 'price_range']))
    weights = {
        'category': 4,
        'subcategory': 4,
        'price_range': 2
    }
    for col in encoded_df.columns:
        if 'category' in col:
            encoded_df[col] *= weights['category']
        elif 'subcategory' in col:
            encoded_df[col] *= weights['subcategory']
        elif 'price_range' in col:
            encoded_df[col] *= weights['price_range']

    return cosine_similarity(encoded_df)

def get_content_based_recommendations(item_id, page):
    global df_item, cosine_sim

    matching_indices = df_item.index[df_item['id'] == int(item_id)].tolist()

    if not matching_indices:
        print(f"Item ID {item_id} không tồn tại trong DataFrame.")
        raise  ValueError(f"Item ID {item_id} không tồn tại trong DataFrame.")
    else:
        idx = matching_indices[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        start = (int(page) - 1) * 4
        end = start + 4
        sim_scores = sim_scores[start:end]  
        item_indices = [i[0] for i in sim_scores]

        return df_item.iloc[item_indices]

def ini_cosine_sim1():
    global df_ratings, user_similarity_df, rating_matrix

    if df_ratings.duplicated(['user_id', 'item_id']).any():
        df_ratings = df_ratings.drop_duplicates(['user_id', 'item_id'])
    rating_matrix = df_ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    rating_matrix_normalized = rating_matrix.copy()
    for column in rating_matrix.columns:
        mask = rating_matrix[column] != -1
        scaled_values = scaler.fit_transform(rating_matrix.loc[mask, [column]]).flatten()
        rating_matrix_normalized.loc[mask, column] = scaled_values
    # rating_matrix = np.abs(rating_matrix_normalized)
    rating_matrix = rating_matrix_normalized

    user_similarity = cosine_similarity(rating_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix.index, columns=rating_matrix.index)

    return user_similarity_df

def get_collaborative_recommendations(user_id, page):
    global user_similarity_df, rating_matrix

    user_id = int(user_id)
    page = int (page)
    
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    similar_users = similar_users.index.tolist()
    items_bought = rating_matrix.loc[similar_users].stack().reset_index()
    items_bought.columns = ['user_id', 'item_id', 'rating']

    user_items = rating_matrix.loc[user_id]
    user_items = user_items[user_items > 0].index.tolist() 
    items_bought = items_bought[~items_bought['item_id'].isin(user_items)]

    start = (page - 1) * 4
    end = start + 4
    paginated_items = items_bought['item_id'].unique()[start:end]

    return paginated_items


@app.route('/recommendation/<id_user>/<id_item>/<page>')
def home(id_user ,id_item, page):
    print("id và page", id_user, id_item, page)

    global df_item
    check_id_content_based = 1
    check_id_collaborative = 1
    try:
        content_based_recommendations = get_content_based_recommendations(id_item, page)
    except Exception as e:
        print("lỗi ở phần content_based",e)
        check_id_content_based = 0
    
    try:
        list_item_id = get_collaborative_recommendations(id_user, page)
        print("danh sách id item", list_item_id)
        collaborative_recommendations = df_item[df_item['id'].isin(list_item_id)]
    except Exception as e:
        print("lỗi ở phần collaborative_recommendations",e)
        check_id_collaborative = 0

    if check_id_collaborative and check_id_content_based:
        final_recommendations = pd.concat([content_based_recommendations, collaborative_recommendations])
    elif check_id_collaborative:
        final_recommendations = collaborative_recommendations
    else:
        final_recommendations = content_based_recommendations

    selected_columns = ['id', 'subcategory', 'name', 'current_price', 'image_url'] 
    filtered_recommendations = final_recommendations[selected_columns] 
    records = filtered_recommendations.to_dict(orient='records')
    return jsonify(records)

if __name__ == '__main__':
    df_item = pd.read_excel(file_path)
    df_ratings = pd.read_excel(file_path1)
    cosine_sim = ini_cosine_sim()
    item_similarity_df = ini_cosine_sim1()
    app.run(host='localhost', port=5000)
