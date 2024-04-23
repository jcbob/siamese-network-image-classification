# siamese-network-image-classification
Repository for fruit &amp; vegetable classification based on siamese network

---
- Date begun: 04.03.2024
- Date ended: -
- Author(s): Patryk Gawłowski, Filip Żmuda, Kamil Tokarek, Jakub Wolski, firma InsERT (Jan Sowa, Krzysztof Raszczuk)
---
## ToDo
##### Priority table
| Priority | Task                                          |
| -------- | --------------------------------------------- |
| 1 | implementacja narzędzi do porównywania modeli |
| 2 | refactoring dwóch notebooków (?)              |
| 3 | szukanie kolejnych zbiorów + odpalenie obecnych modeli na "fruit recognition".|
| 4 | zainteresować się artykułem: [building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 

##### Others
- podpiąć dataset "fruit-recognition" jako bazę danych

- zaimplementowanie metryk porównujących modeli
	- top3/top1 accuracy
	- czy prawidłowy etykieta owocu znajduje się w pierwszej 3-ce zaproponowanych przez model

- funkcja (osobna od modelu), która zwraca wektor z listy do której jest najbliżej wektor podany jako argument
	- użyć euclidean_disctance()

- szukać bazy danych z produktami spożywczymi/warzywami/owocami, które mają też etykiety

- jak zapisywać wagi do pliku

- przerzucanie kodu z notebooków do plików `.py`
	- tworzenie bazy danych - stworzenie tf.data.Dataset
	- moduły pomagające w budowie modelu
	- ocena jakości modelu

