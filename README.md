# SQuAD
### Задача:
По входным контексту и вопросу необходимо найти ответ на данный вопрос в данном контексте.

### Ссылки:
[Сайт соревнования](https://rajpurkar.github.io/SQuAD-explorer/)
[Статья](https://arxiv.org/pdf/1704.00051.pdf), на основе которой была написана модель
Обработанные данные взяты [отсюда](https://github.com/facebookresearch/DrQA)

### Установка

<pre>git clone https://github.com/facebookresearch/DrQA.git</pre>
<pre>cd squad</pre>
<pre>pip3 install -r packages.txt</pre>
Также необходимо установить tensorflow
CPU версия:
<pre>pip3 install tensorflow</pre>
GPU версия:
<pre>pip3 install tensorflow-gpu</pre>
<pre>python3 prepare.py</pre>
