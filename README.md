
[python-cheatsheet](https://github.com/gto76/python-cheatsheet#list)：全面且实用的 Python 备忘录。这个东西特别适合我这个上了年纪的人，比如：忘记怎么用 Python 写正则、要搞个进度条忘记库的名字和基本用法、用 pandas 处理数据忘记方法需要的参数等等。正当我觉得自己需要“回炉重学”的时候发现了这个项目，有了它上面的问题都可以找到拿来即用的代码片段。我又是那个快乐的 Pythoneer 了，示例代码：

```python3
>>> from collections import Counter
>>> colors = ['blue', 'blue', 'blue', 'red', 'red']
>>> counter = Counter(colors)
>>> counter['yellow'] += 1
Counter({'blue': 3, 'red': 2, 'yellow': 1})
>>> counter.most_common()[0]
('blue', 3)
```
