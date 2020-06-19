import numpy as np
import random
import time
from sklearn.svm import SVC # SVMライブラリ
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 線形判別分析ライブラリ
from sklearn.neighbors import KNeighborsClassifier # k近傍法ライブラリ
from sklearn.model_selection import train_test_split # データ分割ライブラリ
from sklearn.model_selection import cross_val_score # 交差検証ライブラリ
from sklearn.model_selection import KFold
from IPython.display import display
import pandas as pd


''' グローバル変数 '''
# GestureDataのファイルパスを取得
# 各ジェスチャのファイルパス
path1 = "gestureData/gesture1.txt"
path2 = "gestureData/gesture2.txt"
path3 = "gestureData/gesture3.txt"
path4 = "gestureData/gesture4.txt"
path5 = "gestureData/gesture5.txt"
path6 = "gestureData/gesture6.txt"
path7 = "gestureData/gesture7.txt"
path8 = "gestureData/gesture8.txt"
path9 = "gestureData/gesture9.txt"
path10 = "gestureData/gesture10.txt"
path11 = "gestureData/gesture11.txt"

pathList = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11]

gesture_name = []
for i in range(11):
    gesture_name.append('動作'+str(i+1))
    
name_list = ["svm_linear", "svm_poly", "svm_rbf", "lda", "knn"]

    
''' getIEMGListメソッド '''
# pathに該当するファイルのデータをリスト型で取得
def getEmgArray2(path):
    with open(path) as f:
            s = f.read()
    # テキストファイルの内容は全て文字列型なので、Numpy配列の機械学習しやすいように変換する
    s_l = s.split('\n')
    #s_l = s.split('Trial')
    del s_l[0:3]
    #print(s_l)

    trialList = []
    for i in range(len(s_l)):
        trial = []
        sample = []
        if 'Trial' in s_l[i]:
            trial=s_l[i+1: i+9]
            for num in range(len(trial)):
                lis = trial[num].split("\t")
                lis.remove('')
                sample.append(lis)    
            trialList.append(sample)
    trialArr = np.array(trialList, dtype=np.int64)
    return trialArr

# 全波整流平滑化
def full_smoothing(arr):
    arr = np.abs(arr)
    sampling = 100 # サンプリング周波数
    cut_off = 4 # カットオフ周波数
    tf = sampling/(cut_off+sampling) # 伝達関数
    iemg_list = []
    for sensor in range(len(arr)):
        iemg = 0
        lis = []
        for sample in range(len(arr[0])):
            iemg = tf*iemg+(1-tf)*arr[sensor][sample]
            lis.append(iemg)
        iemg_list.append(lis)
    iemg_arr = np.array(iemg_list)
    return iemg_arr

# 半波整流平滑化
def half_smoothing(arr):
    arr = np.where(arr<0, 0, arr)
    sampling = 100 # サンプリング周波数
    cut_off = 4 # カットオフ周波数
    tf = sampling/(cut_off+sampling) # 伝達関数
    iemg_list = []
    for sensor in range(len(arr)):
        iemg = 0
        lis = []
        for sample in range(len(arr[0])):
            iemg = tf*iemg+(1-tf)*arr[sensor][sample]
            lis.append(iemg)
        iemg_list.append(lis)
    iemg_arr = np.array(iemg_list)
    return iemg_arr

'''
getIEMGList(rectification='full', feature=False, cut=True)：
11個の動作データを全波整流か半波整流をして平滑化したIEMGデータを返す。処理時間1~3秒。
--------------------------------------------------
返り値::iemgList[gesture][trial][sensor][sample]

gesture：11動作の番号。0~10で指定する。

trial：計測回数の番号。各動作20回計測したから0~19で指定するが、いづれかの動作の0から数えて19番目のデータが計測できていないから
　　　除外したほうがいい。
   
sensor：計測に使用したMyoアームウェアには筋電位センサが８つ付いており、各センサー番号を指定する。0~7で指定する。

sample：サンプリングレート0.001sで2秒間計測したため、cut=Falseの場合はインデックスの数は0~199。
　　　　　cut=Trueで feature=Trueの場合はインデックスの数は0~153。
　　　　　cut=Trueで feature=Falseの場合はインデックスの数は0~150。
--------------------------------------------------
引数::
rectification : full, half
rectification = full の場合、全波整流がされる。
rectificarion = half の場合、半波整流がされる。

feature : True, False
feature = True の場合、特徴抽出処理を行う用のデータ。sampleのデータ数が153になる。
feature = False の場合、特徴抽出処理を行わない用のデータを返す。sampleのデータ数が150になる。

cut : True, False
cut = True の場合、特徴抽出しないデータのとき前方50データをカット。特徴抽出するデータのとき前方47データをカット。
cut = False の場合、カットしない。
'''
def getIEMGList(rectification='full', feature=False, cut=True):
    iemgList = []
    for path in pathList:
        trialList = []
        arr = getEmgArray2(path)
        for trial in range(len(arr)):
            emg_arr = arr[trial]
            
            # 全波整流する場合
            if rectification == 'full':
                rectification_arr = full_smoothing(emg_arr) # 全波整流平滑化したデータ
            # 半波整流する場合
            elif rectification == 'half':
                rectification_arr = half_smoothing(emg_arr) # 半波整流平滑化したデータ
            
            # 立ち上がり部分を削除する場合
            if cut:
                # 特徴抽出処理を行う場合
                if feature:
                    trialList.append(np.delete(rectification_arr, np.s_[0:47], 1).tolist())
                # 特徴抽出処理を行わない場合
                else:
                    trialList.append(np.delete(rectification_arr, np.s_[0:50], 1).tolist())
            # 立ち上がり部分を削除しない場合
            else:
                trialList.append(rectification_arr.tolist())
        iemgList.append(trialList)
    return iemgList




''' featureExtractionListメソッド '''
#特徴抽出
# MAV mean absolute value
def calc_mav(frame):
    frame_length = len(frame)
    return np.sum(np.abs(frame))/frame_length

# MAV1 modified mean absolute value 1
def calc_mav1(frame):
    frame_length = len(frame)
    wn = np.where((0.25*frame_length <= frame) & (frame <= 0.75*frame_length), 1, 0.5)
    mav1 = np.sum(wn*np.abs(frame))/frame_length      
    return mav1

'''
featureExtractionList(featureIEMG)：
getIEMGListで取得した特徴抽出用のデータを引数として渡すと、特徴抽出手法MAVとMAV1を行った特徴量データを返す。
--------------------------------------------------
返り値::
mavIEMG[gesture][trial][sensor][sample]：IEMGをMAVで特徴抽出した特徴量
mav1IEMG[gesture][trial][sensor][sample]：IEMGをMAV1で特徴抽出した特徴量
gesture：11動作の番号。0~10で指定する。

trial：計測回数の番号。各動作20回計測したが20回目のデータが計測ミスを起こしているから除外して、0~18でインデックスを指定。

sensor：計測に使用したMyoアームウェアには筋電位センサが８つ付いており、各センサー番号を指定する。0~7で指定する。

sample：インデックスを0~149で指定する。

--------------------------------------------------
引数::
featureIEMG[gesture][trial][sensor][sample]：getIEMGList関数で取得した特徴抽出用のデータを使用する。
featureIEMG[11][20][8][153]
'''
def featureExtractionList(featureIEMG):
    mavIEMG = []
    mav1IEMG = []
    for gesture in range(11):
        mavTrial = []
        mav1Trial = []
        for trial in range(19):       
            mavSensor = []
            mav1Sensor = []
            for sensor in range(8):
                frame = []
                mav =[]
                mav1 = []
                for sample in range(153):
                    frame.append(featureIEMG[gesture][trial][sensor][sample])

                    if len(frame) == 4:
                        mav.append(calc_mav(np.array(frame)))
                        mav1.append(calc_mav1(np.array(frame)))
                        del frame[0]

                mavSensor.append(mav)
                mav1Sensor.append(mav1)
            mavTrial.append(mavSensor)
            mav1Trial.append(mav1Sensor)
        mavIEMG.append(mavTrial)
        mav1IEMG.append(mav1Trial)
    return mavIEMG, mav1IEMG




''' makeLearnTestDataメソッド '''
'''
makeLearnTestData(iemg, number=2)：
引数iemgのデータを引数numberで指定した学習データ数から、学習用のデータリストとテスト用のデータリスト、データラベルを作成する
--------------------------------------------------
返り値::
learnIEMG[8][150×11×学習データ数]：numberで設定した学習データ数の学習用データリストを作成する。インデックスは150と11とnumberの積。
testIEMG[8][16500]：テストデータは10トライアル分を使用することにした。インデックスは0~16499で指定する。
gesLearn[150×11×学習データ数]：learnIEMGのインデックス数に合わせたデータラベルリストを作成する。
gesTest[16500]：testIEMGのインデックス数に合わせたデータラベルリストを作成する。
--------------------------------------------------
引数::
iemg：iemg[gesture][trial][sensor][sample]の形状のリストを渡す。
number：学習データ数を指定する。指定した学習データ数をもとに学習用データの長さを変更する。

'''
def makeLearnTestData(iemg, number=2):
    random.seed(2)
    index_list = list(range(19))
    learn_index = random.sample(index_list, 9) # 訓練データのインデックスリスト
    test_index = index_list
    for i in learn_index:
        test_index.remove(i) # 未知データのインデックスリスト
    learn_number = random.sample(learn_index, number) # 訓練データのインデックスリスト
    gesture = 1
    gesLearn, gesTest = [], []
    
    learnIEMG = [[],[],[],[],[],[],[],[]]
    testIEMG = [[],[],[],[],[],[],[],[]]
    
    print(learn_number)
    print(test_index)
    
    for gesture in range(11):
        # 訓練データのリスト生成
        for trial in learn_number:
            for sensor in range(8):
                learnIEMG[sensor].extend(iemg[gesture][trial][sensor])

        # 未知データのリスト生成
        for trial in test_index:
            for sensor in range(8):
                testIEMG[sensor].extend(iemg[gesture][trial][sensor])

        if gesture == 0:
            gesLearn = np.full(150*len(learn_number), gesture+1)
            gesTest = np.full(150*len(test_index), gesture+1)
        else:
            gesLearn = np.append(gesLearn, np.full(150*len(learn_number), gesture+1))
            gesTest = np.append(gesTest, np.full(150*len(test_index), gesture+1))
    return learnIEMG, testIEMG, gesLearn, gesTest




''' gestureScoreメソッド '''
def testDataPredictScore(model, trainDataSet, trainLavelSet, testDataSet, testLavelSet):
            model.fit(np.array(trainDataSet), np.array(trainLavelSet))
            start = 0
            end = 1500
            gestureScores = []
            gestureRate = []
            for i in range(11):
                score = model.score(np.array(testDataSet).T[start: end], testLavelSet[start : end])
                
                predictResult = model.predict(np.array(testDataSet).T[start: end])
                rates = []
                for number in range(11):
                    counter = 0
                    for index in range(len(predictResult)):
                        if number+1 == predictResult[index]:
                            counter += 1
                    rates.append(counter/len(predictResult))
                
                #print("動作{} score: {}".format(i, score))
                start += 1500
                end += 1500
                gestureScores.append(score)
                gestureRate.append(rates)
            model_score = model.score(np.array(testDataSet).T, testLavelSet)
            #print("全体 score: {}\n".format(model_score))
            
            return model_score, gestureScores, gestureRate
    
# SVM識別器の生成    
def generate_svm(trainDataSet, trainLavelSet, kernel_str=["linear", "poly", "rbf"]):     
    param_list = [0.001, 0.01, 0.1, 1, 10, 100]
    svm_list = []
    for k in kernel_str:
        print(k)
        best_score = 0
        best_parameters = {}
        # 交差検証を用いたグリッドサーチ
        i = 0
        for gamma in param_list:
            for C in param_list:
                # SVCのインスタンス生成
                svm = SVC(kernel = k, gamma=gamma, C=C, random_state=None)
                
                # 交差検証 パラメータcvで分割方法を設定
                scores, gestureScore = kFoldCrossValidation(svm, trainDataSet, trainLavelSet, cv=5)
                
                # 交差検証による評価値の平均
                score = np.mean(scores)
                
                print("\r交差検証 {}/{}  {}".format(i+1, len(param_list)**2, score), end="")
                i += 1
                if score>best_score:
                    best_score = score
                    best_parameters = {'gamma' : gamma, 'C' : C}
                    best_gestureScore = gestureScore
        print('\n')
        # もっともスコアの高いパラメータで識別器をインスタンス
        svm = SVC(kernel=k, **best_parameters)
        
        # モデルの保存
        svm_list.append(svm)

        print('Best score on validation set: {}'.format(best_score))
        print('Best parameters: {}'.format(best_parameters))
        print('Best gesture score: {}'.format(best_gestureScore))
        print('\n')
       
        
    return svm_list, best_score
    

# 線形判別分析の識別器生成        
def generate_lda(trainDataSet, trainLavelSet): 
    lda = LinearDiscriminantAnalysis()

    # 交差検証
    #scores = cross_val_score(lda, x_train, y_train, cv=8)
    scores, gestureScore = kFoldCrossValidation(lda, trainDataSet, trainLavelSet, cv=5)
    validation_score = np.mean(scores)
    
    print("LDA")
    print("Score on validation set: {}".format(validation_score))
    print('Gesture score: {}'.format(gestureScore))
    
    return lda, validation_score

# k近傍法の識別器生成
def generate_knn(trainDataSet, trainLavelSet):  
    knn = KNeighborsClassifier(n_neighbors = 3)

    # 交差検証
    #scores = cross_val_score(knn, x_train, y_train, cv=8)
    scores, gestureScore = kFoldCrossValidation(knn, trainDataSet, trainLavelSet, cv=5)
    validation_score = np.mean(scores)

    print("knn")
    print("Score on validation set: {}".format(validation_score))
    print('Gesture score: {}'.format(gestureScore))
    
    return knn, validation_score

# k-fold cross validationを実行する。[1650][8]のtrainDataを引数に渡す。
def kFoldCrossValidation(model, trainData, trainLavel, cv=5): 
    trainDataTlist = trainData.tolist()
    # 学習データセットの学習データとラベルを連結
    for i in range(len(trainDataTlist)):
        trainDataTlist[i].append(trainLavel[i])

    # 交差検証するための分割のインスタンス生成
    kf = KFold(n_splits = cv, shuffle = True)
    
    scoreList, gestureScoreList = [], []
    j=1
    for train_index, test_index in kf.split(trainDataTlist):
        trainDataList, testDataList = [], []
        trainLavelList, testLavelList = [],[]
        testGestureData = [[], [], [], [], [], [], [], [], [], [], []]
        # 生成した訓練用データのインデックスから訓練データとラベルデータを分けたリストを生成する
        for i in train_index:
            trainDataList.append(trainDataTlist[i][0:8])
            trainLavelList.append(trainDataTlist[i][8])
        # 生成した評価用データのインデックスから評価データとラベルデータを分けたリストを生成する
        for i in test_index:
            testDataList.append(trainDataTlist[i][0:8])
            testLavelList.append(trainDataTlist[i][8])
            testGestureData[trainDataTlist[i][8]-1].append(trainDataTlist[i][0:8]) # 動作ラベル毎に評価データを分ける

        model.fit(trainDataList, trainLavelList) # 訓練データで識別モデルの作成
        
        scoreList.append(model.score(testDataList, testLavelList)) # 評価データでの全体スコア計算
        
        gestureScore = []
        # 動作スコア計算
        for i in range(11):
            gestureScore.append(model.score(testGestureData[i], [i+1]*len(testGestureData[i]))) # 評価データで各動作のスコア計算
        gestureScoreList.append(gestureScore)
        
        #print('動作スコア: {}'.format(gestureScore))
    #print(np.mean(np.array(scoreList))) # 全体のスコアの平均
    
    gestureScoreAverage = np.mean(np.array(gestureScoreList), axis=0) # 各動作のスコアの平均
    #gestureScorePrint(gestureScoreAverage) # 各動作のスコア表示
    
    return np.array(scoreList), gestureScoreAverage

# 学習用データセットから各識別器を生成する。学習用データセットとラベルセットを引数として渡す。
def generateAllDiscriminater(trainDataSet, trainLavelSet, testDataSet, testLavelSet):
    
    # 識別器の保存リスト
    model_list = []
    
    # 識別器名リスト
    name_list = ["svm_linear", "svm_poly", "svm_rbf", "lda", "knn"]
    
    gesture_name = []
    for i in range(11):
        gesture_name.append('ges'+str(i+1))
    
    # score_list
    val_score_list = []
    test_score_list = []
    gestureScoresList = []
    gestureRateList = []
    
    for method in [generate_svm, generate_lda, generate_knn]:
        model, validation_score = method(trainDataSet, trainLavelSet)
        
        if method == generate_svm:
            model_list = model
            for i in range(3):
                print('{}\n'.format(name_list[i]))
                modelScore, gestureScores, gestureRate = testDataPredictScore(model[i], trainDataSet, trainLavelSet, testDataSet, testLavelSet)
                gestureScoresList.append(gestureScores)
                test_score_list.append(modelScore)
                gestureRateList.append(gestureRate)
                df_gestureRate = pd.DataFrame(gestureRate, gesture_name, gesture_name)
                display(df_gestureRate.round(3))
                print('テストデータの識別精度: {}\n動作スコア: {}\n'.format(modelScore, gestureScores))
        else:
            model_list.append(model)
            modelScore, gestureScores, gestureRate = testDataPredictScore(model, trainDataSet, trainLavelSet, testDataSet, testLavelSet)
            gestureScoresList.append(gestureScores)
            test_score_list.append(modelScore)
            gestureRateList.append(gestureRate)
            df_gestureRate = pd.DataFrame(gestureRate, gesture_name, gesture_name)
            display(df_gestureRate.round(3))
            print('テストデータの識別精度: {}\n動作スコア: {}\n'.format(modelScore, gestureScores))
        val_score_list.append(validation_score)
        
        
    return model_list, val_score_list, name_list, test_score_list, gestureScoresList, gestureRateList
    # model_list, val_score_list, test_score_list, name_list = generate_all_dicriminater(x_trainval, y_trainval)


def gestureScorePrint(gestureScoreAverage):
    print('動作1 {:.2f}, 動作2 {:.2f}, 動作3 {:.2f}, 動作4 {:.2f}, 動作5 {:.2f}, 動作6 {:.2f}, 動作7 {:.2f}, 動作8 {:.2f}, 動作9 {:.2f}, 動作10 {:.2f}, 動作11 {:.2f}'
          .format(gestureScoreAverage[0], gestureScoreAverage[1], gestureScoreAverage[2], gestureScoreAverage[3],
                  gestureScoreAverage[4], gestureScoreAverage[5], gestureScoreAverage[6], gestureScoreAverage[7],
                  gestureScoreAverage[8], gestureScoreAverage[9], gestureScoreAverage[10]))
    

'''
gestureScore(learn, test, learnLavel, testLavel)：
引数で渡された学習データで学習を行い識別器を作成して、引数で渡された未知データのラベルを予測してラベル毎の識別精度とモデルの識別精度を返す。
--------------------------------------------------
返り値::
gesture_score[model][gesture] : 各識別器における動作毎のスコアを保管してある。識別器はsvm_linear、svm_ploy、svm_rbf、LDA、kNNの順に
                                                記録されている。gestureは11動作でインデックス0~10で指定する。

model_score_lis[model] : 各識別器におけるスコアを保管してある。modelは５識別器でインデックス0~4で指定する。
--------------------------------------------------
引数::
learn : makeLearnTestData関数で作成した学習データを渡す。

test : makeLearnTestData関数で作成したテストデータを渡す。

learnLavel : makeLearnTestData関数で作成した学習データの動作ラベルを渡す。

testLavel : makeLearnTestData関数で作成したテストデータの動作ラベルを渡す。
'''
def gestureScore(learn, test, learnLavel, testLavel):

    train = np.array(learn).T
    #model_list, val_score_list, name_list = generate_all_dicriminater(train, learnLavel)
    model_list, val_score_list, name_list, test_score_list, gesture_score, gestureRate = generateAllDiscriminater(train, learnLavel, test, testLavel)
    
    return gesture_score, test_score_list, gestureRate