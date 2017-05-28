from sklearn.datasets import load_svmlight_files
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from fm import FactorizationMachine

fm = FactorizationMachine('/home/ffl/nus/MM/cur_trans/tool/libfm',
                          dim='1,1,8', init_stdev=0.005, iter=200,
                          method='mcmc', out='test_fm.out', task='c',
                          test='a1a.t', train='a1a')
svc = SVC(kernel='rbf', probability=True)
svc_1 = SVC(kernel='linear', probability=True)
# ensemble_model = VotingClassifier(estimators=[('fm', fm), ('svc', svc)],
#                                   voting='soft', weights=[1, 0])
# ensemble_model = VotingClassifier(estimators=[('svc1', svc_1), ('svc', svc)],
#                                   voting='hard', weights=[2, 1])
# ensemble_model = VotingClassifier(estimators=[('fm', fm), ('svc', svc),
#                                               ('svc1', svc_1)],
#                                   voting='hard', weights=[1,1,1])
ensemble_model = VotingClassifier(estimators=[('fm', fm), ('svc', svc),
                                              ('svc1', svc_1)],
                                  voting='soft', weights=[1,1,1])
X_train, y_train, X_test, y_test = load_svmlight_files(('/home/ffl/nus/MM/tencent_ad/a1a',
                                                        '/home/ffl/nus/MM/tencent_ad/a1a.t'))
# train
fm = fm.fit(X_train, y_train)
svc = svc.fit(X_train, y_train)
svc_1 = svc_1.fit(X_train, y_train)
ensemble_model = ensemble_model.fit(X_train, y_train)

pre_fm = fm.predict(X_test)
pre_svc = svc.predict(X_test)
pre_pro_svc = svc.predict_proba(X_test)
# print pre_pro_svc.shape
# print pre_pro_svc
pre_svc_1 = svc_1.predict(X_test)
# print pre_svc_1
pre_ensemble_model = ensemble_model.predict(X_test)
# print pre_ensemble_model
print 'fm:', accuracy_score(y_test, pre_fm)
print 'svc:', accuracy_score(y_test, pre_svc)
print 'svc_1:', accuracy_score(y_test, pre_svc_1)
print 'ensemble:', accuracy_score(y_test, pre_ensemble_model)