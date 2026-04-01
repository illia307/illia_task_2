import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Налаштування сторінки
st.set_page_config(page_title='Екологічна аналітика', page_icon='🍃', layout='wide')
st.title('🍃 Екологічна аналітика: Якість повітря (PM2.5)')

# --- 1. Завантаження даних ---
@st.cache_data
def load_data():
    df = pd.read_csv('eco_data.csv')
    df['Дата'] = pd.to_datetime(df['Дата'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Файл 'eco_data.csv' не знайдено.")
    st.stop()

# --- 2. Карта забруднення ---
st.subheader('🗺️ Карта забруднення (Останні зафіксовані дані)')
st.write("Розмір точок на карті залежить від рівня забруднення PM2.5.")

# Беремо лише останню дату для відображення актуальної ситуації на карті
latest_date = df['Дата'].max()
map_data = df[df['Дата'] == latest_date].copy()

# Додаємо колонку для розміру точок на карті (множимо PM2.5 для наочності)
map_data['size'] = map_data['PM2_5'] * 50 

# Виводимо карту (колір точок червоний, розмір залежить від рівня PM2.5)
st.map(map_data, latitude='lat', longitude='lon', size='size', color='#ff4b4b')

st.divider()

# --- 3. Аналіз трендів ---
st.subheader('📈 Аналіз трендів рівня PM2.5')
st.write("Динаміка зміни якості повітря за обраний період.")

# Перетворюємо таблицю для графіка: рядки - дати, колонки - локації
trend_data = df.pivot(index='Дата', columns='Локація', values='PM2_5')
st.line_chart(trend_data)

st.divider()

# --- 4. Прогноз рівня PM2.5 ---
st.subheader('🔮 Прогноз рівня PM2.5 на наступні 7 днів')
locations = df['Локація'].unique()
selected_loc = st.selectbox('📍 Оберіть датчик (локацію) для побудови прогнозу:', locations)

# Підготовка даних для обраної локації
loc_data = df[df['Локація'] == selected_loc].sort_values('Дата').copy()
min_date = loc_data['Дата'].min()
loc_data['Дні'] = (loc_data['Дата'] - min_date).dt.days

X = loc_data[['Дні']]
y = loc_data['PM2_5']

# Навчання моделі лінійної регресії
model = LinearRegression()
model.fit(X, y)

# Прогнозування на майбутні 7 днів
future_days = 7
max_day = loc_data['Дні'].max()
future_X = pd.DataFrame({'Дні': range(max_day + 1, max_day + 1 + future_days)})
future_predictions = model.predict(future_X)

# Генерація дат для прогнозу
last_date = loc_data['Дата'].max()
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]

# Формування фінальної таблиці для графіка прогнозу
df_hist = loc_data.set_index('Дата')[['PM2_5']].rename(columns={'PM2_5': 'Історичні дані'})
df_future = pd.DataFrame({'Дата': future_dates, 'Прогноз': future_predictions}).set_index('Дата')

# Рівень PM2.5 не може бути від'ємним
df_future['Прогноз'] = df_future['Прогноз'].apply(lambda x: max(0, x))

# Об'єднуємо історичні дані та прогноз
plot_df = df_hist.join(df_future, how='outer')

# Відображення графіка прогнозу
st.line_chart(plot_df, color=["#1f77b4", "#ff7f0e"])

# Текстовий висновок на основі прогнозу
trend_direction = "зростання" if model.coef_[0] > 0 else "зниження"
st.info(f"💡 **Аналіз алгоритму:** На основі історичних даних у локації **{selected_loc}** спостерігається стійка тенденція до **{trend_direction}** рівня забруднення.")
