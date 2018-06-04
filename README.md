# SQuAD
### Задача:
По входным контексту и вопросу необходимо найти ответ на данный вопрос в данном контексте.

### Ссылки:
[Сайт соревнования](https://rajpurkar.github.io/SQuAD-explorer/).  
[Статья](https://arxiv.org/pdf/1704.00051.pdf), на основе которой была написана модель.  
Обработанные данные взяты [отсюда](https://github.com/facebookresearch/DrQA).  

### Установка
'''
git clone https://github.com/facebookresearch/DrQA.git
cd squad
pip3 install -r packages.txt
pip3 install tensorflow # или tensorflow-gpu для GPU версии
python3 prepare.py
'''

### Описание скриптов
  1. train.py - запускает модель на обучение
  2. test.py - считает f1-score для тестового датасета
  3. demo.py - выдает ответ по введенным данным

### Результат
Модель обучалась на 7 эпохах.  
F1-score на тестовом датасете равен 0.43.  
