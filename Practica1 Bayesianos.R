
#####################################################################
#- CARGADO DE DATOS

#Cargado de ambas bases de datos
clickbait_data <- read.delim("C:/Users/jsm01/OneDrive/Escritorio/Cuarto/Métodos Bayesianos/Practica 1/clickbait_data", header=FALSE, comment.char="#")
non_clickbait_data <- read.delim2("C:/Users/jsm01/OneDrive/Escritorio/Cuarto/Métodos Bayesianos/Practica 1/non_clickbait_data", header=FALSE, comment.char="#")

#Creación de la columna clickbait (Variable respuesta)
clickbait_data$clickbait = (rep(1,length(clickbait_data$V1)))
non_clickbait_data$clickbait = (rep(0,length(non_clickbait_data$V1)))

#Juntamos ambas bases de datos 
Datos <- merge(x = clickbait_data, y = non_clickbait_data, all = TRUE)

#Modificamos los nombres de las variables
names(Datos) = c("Titulo", "clickbait")

#Vemos los tipos de nuestras variables
summary(Datos)

#Modificamos la columna clickbait para transformarla en factor
Datos$clickbait <- as.factor(Datos$clickbait)

#Comprobamos la transformación
summary(Datos)


#####################################################################
#- LIMPIEZA DE DATOS

# Cargamos la libreria necesaria para la limpieza
if(!require("tm")){
  install.packages("tm")
}
library("tm")

# Creamos el corpus
corpus <- Corpus(VectorSource(Datos$Titulo))

# Vemos la información contenida en el objeto corpus
inspect(corpus[1:5])

# Ponemos todas las palabras en minuscula aplicando la función tolower
clean_corpus <- tm_map(corpus, tolower)

# Volvemos a ver la información del corpus
inspect(clean_corpus[1:5]) #Vemos que todas las palabras ahora estan en minuscula

# Ahora le vamos a quitar los numeros
clean_corpus2 <- tm_map(clean_corpus, removeNumbers)

# Volvemos a ver la información del corpus
inspect(clean_corpus2[1:5]) #Vemos que han desaparecido los numeros

# Ahora le vamos a quitar los signos de puntuación,
clean_corpus3 <- tm_map(clean_corpus2, removePunctuation)

# Volvemos a ver la información del corpus
inspect(clean_corpus3[1:5]) #Vemos que han desaparecido los signos de puntuacion

# Ahora le vamos a quitar las palabras poco informativas
# Mostramos el ejemplo de algunas

stopwords("en")[1:20]
#Quitamos las palabras de nuestro corpus
clean_corpus4 <- tm_map(clean_corpus3, removeWords,stopwords("en"))

# Volvemos a ver la información del corpus
inspect(clean_corpus4[1:5])

# Quitamos el exceso de espacios en blanco
clean_corpus5 <- tm_map(clean_corpus4, stripWhitespace)

#Volvemos a ver la inforación del corpus
inspect(clean_corpus5[1:5])


#####################################################################
#- VISUALIZACIÓN DE LOS DATOS

# Vamos a hacerlo mediante nubes de palabras

# Primero obtenemos los índices de cada categoría
clickbait_indices <- which(Datos$clickbait == 1)
Noclickbait_indices <- which(Datos$clickbait == 0)

# Cargamos la libreria necesaria para la visualizacion
if(!require("wordcloud")){
  install.packages("wordcloud")
}
library("wordcloud")

# Realizamos la visualización
wordcloud(clean_corpus5[clickbait_indices], min.freq = 200, scale=c(2,.5))
wordcloud(clean_corpus5[Noclickbait_indices], min.freq = 150, scale=c(2,.5))

#####################################################################
#- CONSTRUCCION DEL CLASIFICADOR

# Dividimos los datos en train y test
set.seed(0) #Para que el estudio sea reproducible
observaciones <- dim(Datos)[1]
indices_train <- 1:round(observaciones*0.75)
indices_test <- (round(observaciones*0.75)+1):observaciones

Datos_train <- Datos[indices_train,]
Datos_test <- Datos[indices_test,]

summary(Datos_train)
summary(Datos_test)

# El corpus también lo dividimos
corpus_train <- clean_corpus5[indices_train]
corpus_test <- clean_corpus5[indices_test]

# Conseguimos la matriz esparsa las filas son los documentos y las columnas las palabras

#Construimos la matriz dispersa
Datos_dtm <- DocumentTermMatrix(clean_corpus5)
inspect(Datos_dtm[1:7, 1:9]) #Visualizamos algunos elementos

# Dividimos la matriz en la parte de entrenamiento y test

Datos_dtm_train <- Datos_dtm[indices_train,]
Datos_dtm_test <- Datos_dtm[indices_test,]

# Buscamos las palabras que aparecen 10 o más veces
X_times_words <- findFreqTerms(Datos_dtm_train, 10)
length(X_times_words)

X_times_words[1:10]

# Volvemos a crear las matrices, pero ahora solo con las palabras frecuentes
Datos_dtm_train_frecuente <- DocumentTermMatrix(corpus_train, 
                             control = list(dictionary = X_times_words))
Datos_dtm_test_frecuente <- DocumentTermMatrix(corpus_test, 
                             control = list(dictionary = X_times_words))

# Convertimos los conteos a Si o No
convertir_cuenta <- function(x){
  y <- ifelse(x > 0, 1,0)
  y <- factor(y,levels = c(0,1), labels = c("No", "Si"))
  y
}
# Ahora se lo aplicamos a las matrices
# Primero la train
Datos_dtm_train_frecuente2 <- apply(Datos_dtm_train_frecuente,2,convertir_cuenta)

# Luego la test
Datos_dtm_test_frecuente2 <- apply(Datos_dtm_test_frecuente,2,convertir_cuenta)

Datos_dtm_train_frecuente2[1:7, 1:9]
Datos_dtm_test_frecuente2[1:7, 1:9]

#####################################################################
#- CONSTRUCCION DEL CLASIFICADOR BAYES INGENUO FRECUENTISTA

# Instalamos la libreria necesaria
if(!require("e1071")){
  install.packages("e1071")
}
library("e1071")

# Construimos el clasificador
clasificador_frecuentista <- naiveBayes(Datos_dtm_train_frecuente2, 
                                        Datos_train$clickbait,0)
class(clasificador_frecuentista)

# Calculamos como funciona con la muestra de entrenamiento

t <- proc.time() # Inicia el cronómetro
predicciones_entrenamiento_frecuentista <- predict(clasificador_frecuentista, 
                                                   newdata = Datos_dtm_train_frecuente2)
proc.time()-t    # Detiene el cronómetro

(confusion_matrix_train_frecuentista <- table(predicciones_entrenamiento_frecuentista, 
                                              Datos_train$clickbait))

(accuracy_train_frecuentista <- sum(diag(confusion_matrix_train_frecuentista))/dim(Datos_train)[1])

# Ahora lo vemos con los datos de test
t <- proc.time() # Inicia el cronómetro
predicciones_test_frecuentista <- predict(clasificador_frecuentista, 
                                          newdata = Datos_dtm_test_frecuente2)
proc.time()-t    # Detiene el cronómetro

(confusion_matrix_test_frecuentista <- table(predicciones_test_frecuentista, 
                                             Datos_test$clickbait))

(accuracy_test_frecuentista <- sum(diag(confusion_matrix_test_frecuentista))/dim(Datos_test)[1])


#####################################################################
#- CONSTRUCCION DEL CLASIFICADOR BAYES INGENUO CON DISTRIBUCIONES A PRIORI

# Construimos el clasificador
Clasificador_bayesiano <- naiveBayes(Datos_dtm_train_frecuente2, Datos_train$clickbait, laplace = 1)
class(Clasificador_bayesiano)

# Calculamos como funciona con la muestra de entrenamiento

t <- proc.time() # Inicia el cronómetro
predicciones_entrenamiento_bayesiano <- predict(Clasificador_bayesiano, newdata = Datos_dtm_train_frecuente2)
proc.time()-t    # Detiene el cronómetro

(confusion_matrix_train_bayesiano <- table(predicciones_entrenamiento_bayesiano, 
                                           Datos_train$clickbait))

(accuracy_train_bayesiano <- sum(diag(confusion_matrix_train_bayesiano))/dim(Datos_train)[1])

# Ahora lo vemos con los datos de test
t <- proc.time() # Inicia el cronómetro
predicciones_test_bayesiano <- predict(Clasificador_bayesiano, newdata = Datos_dtm_test_frecuente2)
proc.time()-t    # Detiene el cronómetro

(confusion_matrix_test_bayesiano <- table(predicciones_test_bayesiano, Datos_test$clickbait))

(accuracy_test_bayesiano <- sum(diag(confusion_matrix_test_bayesiano))/dim(Datos_test)[1])


#####################################################################
#- COMPARACION RESULTADOS

precisiones <- matrix(nrow = 2, data = c(accuracy_train_frecuentista,accuracy_train_bayesiano, accuracy_test_frecuentista, accuracy_test_bayesiano), byrow = TRUE)

precisiones <- as.data.frame(precisiones)

names(precisiones) <- c("Frecuentista", "Bayesiano")
row.names(precisiones) <- c("Train", "Test")

precisiones

#Vemos que el bayesiano mejora al frecuentista en ambas comparaciones