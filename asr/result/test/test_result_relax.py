import pandas as pd
import numpy as np


df = pd.DataFrame(np.random.randn(15).reshape(5, 3))

print(df)
df_styled = df.style.format('{:,.2f}').set_properties(**{
    'font': 'Arial', 'color': 'red', 'text-align': 'center'})
df_styled.set_table_styles([{'props': [
    ('background-color', 'MintCream'), ('color', 'blue'), ('font', 'Arial'),
    ('font-weight', 'bold')]}])
df_styled.set_table_attributes('border="1"')

with open('test.html', 'w') as f:
    f.write(df_styled.render())