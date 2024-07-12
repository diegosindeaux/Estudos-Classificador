Desafio realizado para testar os conhecimentos de machine learning.

**What steps would you take to solve this problem? Please describe as completely and clearly as possible all the steps that you see as essential for solving the problem.**
1. Primeiramente, tentaria visualizar os dados e ver se de cara, conseguiria encontrar alguma relação entre os dados e o problema que estamos enfrentando.
2. Trataria os dados de maneira que pudesse utilizar bibliotecas para analisar os dados de forma mais prática.
4. Pensaria em como seria minha divisão de dados e como eu poderia fazer para realmente ter um caso de uso de teste.
5. Criaria um modelo para ver, de cara, qual seria a resposta que eu teria com ele, e, a partir daí, tentar entender em que pontos seria necessário eu mexer.
6. Trabalharia nos pontos que observei necessários.
7. Criaria um novo modelo, dessa vez, com todos os pontos trabalhados.
8. Colocaria pra teste com um caso de uso real.

**Which technical data science metric would you use to solve this challenge? Ex: absolute error, rmse, etc.**
1. Recall -> Uma vez que, caso o meu caminhão vá para a oficina, eu gastaria $ 500,00, eu tento evitar ao máximo que ele vá.
2. Matriz de confusão -> Para ter uma noção de como os gastos estão ocorrendo e como eu posso melhorar tudo.

**Which business metric  would you use to solve the challenge?** <br>
Controle de custos, uma vez que foi o motivo de problema ter sido apresentado.

**How do technical metrics relate to the business metrics?** <br>
Elas se relacionam por, a partir do momento que eu tenho um controle maior de porcentagem que os fenômenos ocorrem, eu consigo fazer uma predição média de quanto mais ou menos seria o meu gasto em casos futuros, dessa maneira, o planejamento se torna completamente mais confiável.

**What types of analyzes would you like to perform on the customer database?** <br>
Correlação entre os dados e análises de trend, para que pudesse, posteriormente, realizar analises preditivas.

**What techniques would you use to reduce the dimensionality of the problem?** <br>
PCA

**What techniques would you use to select variables for your predictive model?** <br>
Utilizaria a importância das features em RandomForest.

**What predictive models would you use or test for this problem? Please indicate at least 3.** <br>
1. Random Forest
2. KNN
3. SVM

**How would you rate which of the trained models is the best?** <br>
Se fosse uma análise entre modelos já prontos, usaria as métricas de data science, porém, vendo sem nada pronto, preferiria o Random Forest pelo tamanho do dataset e pelo alto número de valores faltantes.
