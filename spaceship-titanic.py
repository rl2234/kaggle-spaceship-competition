import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


## Preprocessing function: fills missing values, encodes categorical features, extracts complex features, converts types to integers
def prep(data, is_train = True, encoder = None):
    data = pd.read_csv(data)
    
    # if testing data, the passenger ids are extracted for submission purpose
    passenger_ids = None if is_train else data['PassengerId'] 

    # drop columns not useful for modeling
    data.drop(columns = ['PassengerId', 'Name'], inplace = True, errors = 'ignore')

    # fill missing numerical values
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in num_cols:
        data[col].fillna(data[col].median(), inplace = True)

    # fill missing categorical values
    cat_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
    for col in cat_cols:
            data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown', inplace = True)

    # extract Cabin info 
    cabin_split = data['Cabin'].str.split('/', expand = True)
    data['Deck'] = cabin_split[0]
    data['Num'] = cabin_split[1]
    data['Side'] = cabin_split[2]
    data.drop(columns = ['Cabin'], inplace = True)
    
    # clean and convert 'Num' to integer safely
    data['Num'] = data['Num'].fillna('0')
    data['Num'] = data['Num'].replace('', '0').astype(int)

    # Convert remaining columns to integers
    rem_cols = ['CryoSleep', 'VIP', 'Age']
    for col in rem_cols:
        data[col] = data[col].astype(int)

    # handle 'Transported' for training
    if is_train:
        data['Transported'] = data['Transported'].astype(int)

    # one-hot encode categorical columns
    cats_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
    if is_train:
        encoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
        encoded_array = encoder.fit_transform(data[cats_cols])
    else:
        encoded_array = encoder.transform(data[cats_cols])

    encoded_df = pd.DataFrame(encoded_array, columns = encoder.get_feature_names_out(cats_cols), index = data.index)
    data = pd.concat([data, encoded_df], axis = 1)
    data.drop(columns = cats_cols, inplace = True)

    if is_train:
        return data, encoder
    else:
        return data, passenger_ids

# prepare training data
train_data, encoder = prep('train.csv', is_train=True)
X = train_data.drop(columns = ['Transported'])
y = train_data['Transported']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# train model
clf = GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.2, max_depth = 5, random_state = 42)
clf.fit(X_train, y_train)

# preprocess test set using fitted encoder
test_data, passenger_ids = prep('test.csv', is_train = False, encoder = encoder)

# predict
test_preds = clf.predict(test_data).astype(bool)

# format and save submission
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': test_preds
})
submission.to_csv('submission.csv', index = False)

accuracy_score = accuracy_score(y_test, clf.predict(X_test))
print(f"Test accuracy: {accuracy_score}")