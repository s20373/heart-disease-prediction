import pickle
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

filename = 'model.sv'

def save_model():
    train = pd.read_csv("DSP_13.csv", sep=";")
    train = train.fillna(train.mean())
    X = train.drop('zdrowie', axis=1)
    y = train['zdrowie']

    X_train, X_test, y_train, y_test = train_test_split(train.drop('zdrowie', axis=1),
                                                        train['zdrowie'], test_size=0.2,
                                                        random_state=101)

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pickle.dump(model, open(filename, 'wb'))


def main():
    # save_model()

    model = pickle.load(open(filename, 'rb'))

    st.set_page_config(page_title="chore serce? sprawdź!")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image(
        "https://www.shutterstock.com/shutterstock/photos/1986553997/display_1500/stock-photo-portrait-of-serious-senior-doctor-looking-at-x-ray-image-isolated-on-white-background-1986553997.jpg")

    with overview:
        st.title("chore serce? sprawdź!")

    with left:
        symptoms_slider = st.slider("Objawy:", value=3, min_value=1, max_value=5)
        age_slider = st.slider("Wiek:", value=60, min_value=1, max_value=77)
        comorbidities_slider = st.slider("Choroby współistniejące:", value=2, min_value=0, max_value=5)

    with right:
        height_slider = st.slider("Wzrost:", value=180, min_value=159, max_value=200)
        medicine_slider = st.slider("Leki:", value=2, min_value=1, max_value=4)

    data = [[symptoms_slider, age_slider, comorbidities_slider, height_slider, medicine_slider]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba ma szansę na chorobę serca?")
        st.subheader(("Jeszcze jak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == '__main__':
    main()
