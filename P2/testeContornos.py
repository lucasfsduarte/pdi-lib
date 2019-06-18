from lib.lib import *

# Carrega os labels do conjunto de dados (A-Z)
def retornaLabelsAlfabeto():
    lista = []
    with open("CSV/labels.csv", "r") as file:
        for linha in file:
            lista.append([linha[:1]])
    return np.ravel(lista)

# Cria o classificador, treina e o retorna.
def criaClassificador():

    file = open("CSV/dados.csv", "r")

    df = pd.read_csv(file, squeeze=True, header=None)

    # Convertendo o conteúdo do dataframe para valores decimais;
    df.infer_objects().dtypes

    pca = PCA(n_components=0.90, whiten=True)

    # Conduct PCA

    X = pca.fit_transform(df)

    y = retornaLabelsAlfabeto()

    print( "[STATUS] Criando classificador...")
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                             algorithm="SAMME.R",
                             n_estimators=500)

    print( "[STATUS] Ajustando os dados aos modelos..")
    bdt.fit(X, y) #dados, labels

    # Gerando a matriz de confusão
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    # Plotar em um gráfico
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, y_pred, classes=y_test,
                      title='Confusion matrix, without normalization')
    # plot_confusion_matrix(y_test, y_pred, classes=y_test, normalize=True,
    #                   title='Normalized confusion matrix')
    # plt.show()


    # Exibir matriz de confusão no terminal
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    return bdt, pca

def videoLive():
    bdt, pca = criaClassificador()

    frameData = list()

    key = -1

    for i in range(19): # Esse numero significa a quantidade de imagens que tem no repositorio

        nomeImg = "contornos" + str("_%s.png" % i)
        print(nomeImg)
        nomeDiretorio = "contornosTeste/"
        frame = cv.imread(nomeDiretorio + nomeImg)

        frame = cv.resize(frame, (400,400))
        frameData = vetorDadosPCA(frame, pca)

        predicao = bdt.predict(frameData)

        frame = cv.flip(frame, 1) # Inverte a imagem em 180º
        insereLetraImagem(frame, str(predicao[0]))

        cv.imshow('Imagem processada', frame)
        cv.waitKey(1000)

    cv.destroyAllWindows()

def vetorDadosPCA(frame, pca):

    # Testes de classificação
    # frameData = momentoHuWebCam(frame)
    frameData = momentoZernikeWebCam(frame)

    frameData = frameData.replace('  ', ' ')
    frameData = frameData.split(' ')

    # Remover item vazio '' inserido incorretamente
    while(True):
        try:
            frameData.remove('')
        except:
            break

    frameData = list(map(float, frameData))

    frameData = np.array(frameData)

    frameData = pca.transform(frameData.reshape(1,-1))

    return frameData

def insereLetraImagem(frame, letra):

    font                   = 4
    bottomLeftCornerOfText = (340,380)
    fontScale              = 2
    fontColor              = (255,0,0)
    lineType               = 2

    cv.putText(frame,letra.upper(),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    return

videoLive()
