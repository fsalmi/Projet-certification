# Importing the libraries
from flask import render_template,url_for, redirect, flash, session
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle,json
import ast
import sklearn
import os,sys
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging
from flask import g
import datetime

model_xgboost = pickle.load(open(os.path.join(sys.path[0],'model_best_xgboost.pkl'),'rb'))


app = Flask(__name__)
app.secret_key = "super secret key"


df=pd.read_csv('dataset_trois_users_new2.csv')
df.drop(['Mail_not_clean_ancien','Subject_not_clean_ancien'],axis=1,inplace=True)

df_affichage=df[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
df_affichage=df_affichage.sort_values(by="Date",ascending=False)

liste_trois_user_user=['User','cash-m','corman-s','ermis-f']
liste_trois_user=['User','michelle.cash@enron.com','shelley.corman@enron.com','frank.ermis@enron.com']
liste_trois_noms=['User','Michelle Cash','Shelley Corman','Frank Ermis']

liste_adresses_mails=['jeff.dasovich@enron.com','daren.farmer@enron.com','sally.beck@enron.com','larry.campbell@enron.com','don.baughman@enron.com','lynn.blair@enron.com','rick.buy@enron.com','pete.davis@enron.com','eric.bass@enron.com','drew.fossum@enron.com','john.arnold@enron.com','michelle.cash@enron.com','shelley.corman@enron.com','frank.ermis@enron.com','tom.donohoe@enron.com','robert.benson@enron.com','lindy.donoho@enron.com','james.derrick@enron.com','dana.davis@enron.com','david.delainey@enron.com','mike.carson@enron.com','f..brawner@enron.com','monika.causholli@enron.com','chris.dorland@enron.com','m..forney@enron.com','mark.fisher@enron.com','k..allen@enron.com','susan.bailey@enron.com','lisa.gang@enron.com','rob.gay@enron.com','sean.crandall@enron.com','harry.arora@enron.com','martin.cuilla@enron.com','robert.badeer@enron.com','stacy.dickson@enron.com']
liste_noms=['Jeff Dasovich','Daren Farmer','Sally Beck','Larry Campbell','Don Baughman','Lynn Blair','Rick Buy','Pete Davis','Eric Bass','Drew Fossum','John Arnold','Michelle Cash','Shelley Corman','Frank Ermis','Tom Donohoe','Robert Benson','Lindy Donoho','James Derrick','Dana Davis','David Delainey','Mike Carson','F. Brawner','Monika Causholli','Chris Dorland','M. Forney','Mark Fisher','Phillip Allen','Susan Bailey','Lisa Gang','Rob Gay','Sean Crandall','Harry Arora','Martin Cuilla','Robert Badeer','Stacy Dickson']
liste_users=['dasovich-j','farmer-d','beck-s','campbell-l','baughman-d','blair-l','buy-r','dean-c','bass-e','fossum-d','arnold-j','cash-m','corman-s','ermis-f','donohoe-t','benson-r','donoho-l','derrick-j','davis-d','delainey-d','carson-m','brawner-s','causholli-m','dorland-c','forney-j','fischer-m','allen-p','bailey-s','gang-l','gay-r','crandell-s','arora-h','cuilla-m','badeer-r','dickson-s']
@app.template_filter('nl2br')
def nl2br(s):
    return s.replace("\n", "<br />")
#Home page
@app.route('/', methods=['GET','POST'])
def home():
    session['Est_passe_par_proba']=False
    if os.path.isfile('trash.csv'):
        os.remove('trash.csv')
    if os.path.isfile('Greenbox.csv'):
        os.remove('Greenbox.csv')
    selected_u=liste_trois_user[0]
    return render_template('Home2.html',users_dropdown = liste_trois_user,selected_u=selected_u)

#user changing in dropdown handling 
@app.route('/_affiche_user', methods=['GET','POST'])
def affiche_user():
    session['Est_passe_par_proba']=False
    if request.method == 'POST':
        session['Click_bonne_pratique']=False
        session['Click_clean_mail']=False
        session['Est_passe_par_bonne_pratique']=False
        selected_u = request.form.get('comp_select')
        session['my_user'] = selected_u
        if os.path.isfile('trash.csv'):
            trash=pd.read_csv('trash.csv')
            if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
                trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
                df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
                emails_to=df2[df2.To==selected_u]
                emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
                emails_From=df2[df2.From==selected_u]
                emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
                len_dataset_trash=len(trash_user)
                

            else:
                emails_to=df_affichage[df_affichage.To==selected_u]
                emails_From=df[df.From==selected_u]
                emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
                len_dataset_trash=0
        else:
            emails_to=df_affichage[df_affichage.To==selected_u]
            emails_From=df[df.From==selected_u]
            emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
            len_dataset_trash=0

        

        if os.path.isfile('Greenbox.csv'):
            Greenbox_pd=pd.read_csv('Greenbox.csv')
            if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
                len_dataset_greenbox=session.get('len_greenbox_estime', 0)
            else:
                len_dataset_greenbox=0
        else:
            len_dataset_greenbox=0

        nom=liste_trois_noms[liste_trois_user.index(selected_u)]
        emails=emails_to
        len_dataset_to=len(emails)
        len_dataset_from=len(emails_From)
        nbre_mails = emails.shape[0]
        deb_nbre = 0
        fin_nbre = 19
        emails = emails[deb_nbre:fin_nbre+1]
        session['len_dataset']=len(emails_to)+len(emails_From)

        return render_template('Inbox_page.html', nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

@app.route('/Inbox_page', methods=['GET','POST'])
def affiche_inbox():
	selected_u = session.get('my_user', None)
	if os.path.isfile('trash.csv'):
		trash=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
			trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
			emails_to=df2[df2.To==selected_u]
			emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
			emails_From=df2[df2.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=len(trash_user)
			

		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0
	else:
		emails_to=df_affichage[df_affichage.To==selected_u]
		emails_From=df[df.From==selected_u]
		emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
		len_dataset_trash=0

	

	if os.path.isfile('Greenbox.csv'):
		Greenbox_pd=pd.read_csv('Greenbox.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
			len_dataset_greenbox=session.get('len_greenbox_estime', 0)
		else:
			len_dataset_greenbox=0
	else:
		len_dataset_greenbox=0

	nom=liste_trois_noms[liste_trois_user.index(selected_u)]
	emails=emails_to
	len_dataset_to=len(emails)
	len_dataset_from=len(emails_From)
	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	
	return render_template('Inbox_page.html', nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

@app.route('/Sentbox_page', methods=['GET','POST'])
def affiche_sentbox():
	selected_u = session.get('my_user', None)
	
	if os.path.isfile('trash.csv'):
		trash=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
			trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
			emails_to=df2[df2.To==selected_u]
			emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
			emails_From=df2[df2.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=len(trash_user)

		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0
	else:
		emails_to=df_affichage[df_affichage.To==selected_u]
		emails_From=df[df.From==selected_u]
		emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
		len_dataset_trash=0

	len_dataset_to=len(emails_to)
	len_dataset_from=len(emails_From)
	emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
	emails=emails_From
	
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]

	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	if os.path.isfile('Greenbox.csv'):
		Greenbox_pd=pd.read_csv('Greenbox.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
			len_dataset_greenbox=session.get('len_greenbox_estime', 0)
		else:
			len_dataset_greenbox=0
	else:
		len_dataset_greenbox=0
	return render_template('Sentbox_page.html', nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


@app.route('/Greenbox_page', methods=['GET','POST'])
def affiche_greenbox():
	#session['Est_passe_par_proba']=False
	session['Est_passe_par_greenbox']=True
	selected_u = session.get('my_user', None)
	if os.path.isfile('trash.csv'):
		trash=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
			trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
			len_dataset_trash=len(trash_user)
			df3=df2.copy()

		else:
			len_dataset_trash=0
			df3=df.copy()
	else:
		len_dataset_trash=0
		df3=df.copy()
 
	#Pour le From
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]
	emails_to=df3[df3.To==selected_u]
	emails_From=df3[df3.From==selected_u]
	emails_From.drop_duplicates(subset=['Indice_ancien'],keep='first',inplace=True)
	len_dataset_from=len(emails_From)
	len_dataset_to=len(emails_to)
	dataset_from=df3[df3.From==selected_u]
	emails_labelise_from=dataset_from[(dataset_from.Thread!=1) & (dataset_from.email_automatique_supprime_from!=1) & \
                   (dataset_from.email_automatique_supprime_to!=1) & (dataset_from['Start Date']!=1) & \
                   (dataset_from['Proposed agenda']!=1) & (dataset_from['TRANSSTATUS']!=1) & \
                   (dataset_from['CALENDAR ENRTY']!=1) & (dataset_from['HourAhead']!=1) &\
                   (dataset_from['Autohedge']!=1) & (dataset_from['AutoReply']!=1) & \
                   (dataset_from['doctype_html']!=1) & (dataset_from['Auto Reply']!=1) & \
                   (dataset_from['From_newsletter']!=1) & (dataset_from['FREE']!=1) & \
                   (dataset_from['newsletter_suppressed']!=1) & (dataset_from['enron_mentions']!=1) & \
                   (dataset_from['unsubscribe']!=1)]
	emails_labelise_from.drop(['communities','Thread','Communication','page','name','email_automatique_supprime_from',\
                     'email_automatique_supprime_to','Start Date','Proposed agenda','TRANSSTATUS','CALENDAR ENRTY','HourAhead','Autohedge',\
                     'AutoReply','doctype_html','Auto Reply','From_newsletter','FREE','newsletter_suppressed','enron_mentions','unsubscribe'], axis=1,inplace=True)

	X_from=emails_labelise_from.drop(['importance','indice','Cc','X-bcc', 'Mail_not_clean','From_ancien','To_ancien','Subject_not_clean','Subject','Mail','Date','From','To','Indice_ancien','user'],axis=1)
	y_from=emails_labelise_from.importance
	y_pred_from=model_xgboost.predict(X_from)
	#print(X_from['Subject_not_clean_ancien'].dtypes) 
	y_pred_proba_from=model_xgboost.predict_proba(X_from)[:, 1]*100
	X_from['y_pred_from']=y_pred_from
	X_from['y_pred_proba_from']=y_pred_proba_from
	X_from_total=emails_labelise_from.merge(X_from, left_index=True, right_index=True)
	#X_from_fin=X_from_total.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','user','From_ancien','To_ancien','To'])[['y_pred_from','y_pred_proba_from','importance']].mean().reset_index()
	X_from_fin=X_from_total.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','Cc'])[['y_pred_from','y_pred_proba_from','importance']].mean().reset_index()

	X_from_fin.y_pred_from.replace(0.5, 0,inplace=True)
	X_from_fin.y_pred_from=X_from_fin.y_pred_from.astype(int)
	#X_from_fin.fillna(0,inplace=True)
	#X_from_fin.y_pred_proba_from.replace(np.nan , 0,inplace=True)
	#X_from_fin[np.isnan(X_from_fin)] = 0
	X_from_fin.y_pred_proba_from=X_from_fin.y_pred_proba_from.astype(int)
	X_from_supprime=X_from_fin[X_from_fin.y_pred_from==1]
	X_from_supprime_affiche=X_from_supprime[['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','y_pred_from','y_pred_proba_from','importance','Cc']]
	len_X_to_supprime=len(X_from_supprime)
	
	#Pour le To 
	dataset_to=df3[df3.To==selected_u]
	emails_labelise_to=dataset_to[(dataset_to.Thread!=1) & (dataset_to.email_automatique_supprime_from!=1) & \
                   (dataset_to.email_automatique_supprime_to!=1) & (dataset_to['Start Date']!=1) & \
                   (dataset_to['Proposed agenda']!=1) & (dataset_to['TRANSSTATUS']!=1) & \
                   (dataset_to['CALENDAR ENRTY']!=1) & (dataset_to['HourAhead']!=1) &\
                   (dataset_to['Autohedge']!=1) & (dataset_to['AutoReply']!=1) & \
                   (dataset_to['doctype_html']!=1) & (dataset_to['Auto Reply']!=1) & \
                   (dataset_to['From_newsletter']!=1) & (dataset_to['FREE']!=1) & \
                   (dataset_to['newsletter_suppressed']!=1) & (dataset_to['enron_mentions']!=1) & \
                   (dataset_to['unsubscribe']!=1)]
	emails_labelise_to.drop(['communities','Thread','Communication','page','name','email_automatique_supprime_from',\
                     'email_automatique_supprime_to','Start Date','Proposed agenda','TRANSSTATUS','CALENDAR ENRTY','HourAhead','Autohedge',\
                     'AutoReply','doctype_html','Auto Reply','From_newsletter','FREE','newsletter_suppressed','enron_mentions','unsubscribe'], axis=1,inplace=True)

	X_to=emails_labelise_to.drop(['importance','indice','Cc','X-bcc', 'Mail_not_clean','From_ancien','To_ancien','Subject_not_clean','Subject','Mail','Date','From','To','Indice_ancien','user'],axis=1)
	y_to=emails_labelise_to.importance
	y_pred_to=model_xgboost.predict(X_to)
	y_pred_proba_to=model_xgboost.predict_proba(X_to)[:, 1]*100
	X_to['y_pred_to']=y_pred_to
	X_to['y_pred_proba_to']=y_pred_proba_to
	X_to_total=emails_labelise_to.merge(X_to, left_index=True, right_index=True)
	X_to_fin=X_to_total.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','Cc'])[['y_pred_to','y_pred_proba_to','importance']].mean().reset_index()
	X_to_fin.y_pred_to.replace(0.5, 0,inplace=True)
	X_to_fin.y_pred_to=X_to_fin.y_pred_to.astype(int)
	#X_to_fin.fillna(0,inplace=True)
	#X_to_fin.y_pred_proba_to.replace(np.nan , 0,inplace=True)
	#X_to_fin[np.isnan(X_to_fin)] = 0
	X_to_fin.y_pred_proba_to=X_to_fin.y_pred_proba_to.astype(int)
	X_to_supprime=X_to_fin[X_to_fin.y_pred_to==1]
	X_to_supprime_affiche=X_to_supprime[['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','y_pred_to','y_pred_proba_to','importance','Cc']]
	len_X_to_supprime=len(X_to_supprime)

	X_total_supprime=pd.concat([X_from_supprime_affiche,X_to_supprime_affiche], axis=0, ignore_index=True)
	emails=X_total_supprime
	emails.fillna(0,inplace=True)
	emails['Pred_total']=emails['y_pred_proba_to']+emails['y_pred_proba_from']
	#data=emails.to_msgpack()
	emails['user']=liste_trois_user_user[liste_trois_user.index(selected_u)]
	if request.method == "POST":
		req = request.form['sort_by']
		if req=='Date':
			emails=emails.sort_values(by="Date",ascending=False)
			session['Est_passe_par_proba']=False
		elif req=='Proba':
			emails=emails.sort_values(by="Pred_total",ascending=False)
			session['Est_passe_par_proba']=True
		else:
			emails=emails.sort_values(by="Pred_total",ascending=False)
			session['Est_passe_par_proba']=True
		print('Check ici',req)
	print("session.get('Est_passe_par_proba')",session.get('Est_passe_par_proba'))
	if session.get('Est_passe_par_proba')==True:
		emails=emails.sort_values(by="Pred_total",ascending=False)
	emails.to_csv('Greenbox.csv',index=False)
	emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien', 'importance', 'y_pred_from', 'y_pred_proba_from', 'y_pred_proba_to','y_pred_to','Pred_total','Indice_ancien']]
	session['len_greenbox_estime']=len(emails)
	len_dataset_greenbox=len(emails)
	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	
	return render_template('Greenbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


@app.route('/Bonne_pratique_page', methods=['GET','POST'])
def affiche_bonnepratique():
	session['Click_bonne_pratique']=True
	selected_u = session.get('my_user', None)
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]

	if os.path.isfile('trash.csv'):
		trash=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
			trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
			len_dataset_trash=len(trash_user)

		else:
			df2=df.copy()
			len_dataset_trash=0
	else:
		df2=df.copy()
		len_dataset_trash=0


	emails_to=df2[df2.To==selected_u]
	emails_From=df2[df2.From==selected_u]
	emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
	emails=pd.concat([emails_to,emails_From], axis=0, ignore_index=True)
	print('Check len emails',len(emails))

	dataset_supp_features=emails[(emails.email_automatique_supprime_from==1) | (emails.email_automatique_supprime_to==1) | (emails['Start Date']==1) | (emails['Proposed agenda']==1) | (emails['TRANSSTATUS']==1) | (emails['CALENDAR ENRTY']==1) | (emails['HourAhead']==1) |(emails['Autohedge']==1) | (emails['AutoReply']==1) | (emails['doctype_html']==1) | (emails['Auto Reply']==1) | (emails['From_newsletter']==1) | (emails['FREE']==1)|(emails['newsletter_suppressed']==1) | (emails['enron_mentions']==1) | (emails['unsubscribe']==1)]
	dataset_supp_tot=emails[(emails.Thread==1)|(emails.email_automatique_supprime_from==1)|(emails.email_automatique_supprime_to==1)|(emails['Start Date']==1)|(emails['Proposed agenda']==1)|(emails['TRANSSTATUS']==1)|(emails['CALENDAR ENRTY']==1)|(emails['HourAhead']==1)|(emails['Autohedge']==1)|(emails['AutoReply']==1)|(emails['doctype_html']==1)|(emails['Auto Reply']==1)|(emails['From_newsletter']==1)|(emails['FREE']==1)|(emails['newsletter_suppressed']==1)|(emails['enron_mentions']==1)|(emails['unsubscribe']==1)]
	dataset_supp_thread=emails[(emails.Thread==1)]

	emails_to= emails_to[~emails_to.Indice_ancien.isin(dataset_supp_tot.Indice_ancien)]
	emails_From= emails_From[~emails_From.Indice_ancien.isin(dataset_supp_tot.Indice_ancien)]

	len_dataset_to=len(emails_to)
	len_dataset_from=len(emails_From)
	session['len_dataset_thread'] = len(dataset_supp_thread)
	session['len_dataset_supp_features'] = len(dataset_supp_features)
	session['len_dataset_total_supp'] = len(dataset_supp_tot)
	len_dataset_trash=len(dataset_supp_tot)
	emails=dataset_supp_tot[['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','Cc']]
	emails['user']=liste_trois_user_user[liste_trois_user.index(selected_u)]


	if os.path.isfile('trash.csv'):
		trash_user=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash_user.user.values:
			emails.to_csv('trash.csv', mode='a', header=False,index=False)
			session['Est_passe_par_bonne_pratique']=True
			emails=pd.read_csv('trash.csv')
			emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','Indice_ancien']]
			len_dataset_trash=len(emails)
		else:
			emails.to_csv('trash.csv',index=False)
			session['Est_passe_par_bonne_pratique']=True
			len_dataset_trash=len(emails)
	else:
		emails.to_csv('trash.csv',index=False)
		session['Est_passe_par_bonne_pratique']=True
		len_dataset_trash=len(emails)


	session['len_dataset_trash']=len_dataset_trash
	if os.path.isfile('Greenbox.csv'):
		Greenbox_pd=pd.read_csv('Greenbox.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
			len_dataset_greenbox=session.get('len_greenbox_estime', 0)
		else:
			len_dataset_greenbox=0
	else:
		len_dataset_greenbox=0
	
	emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	
	return render_template('Trashbox_page.html', nom=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

@app.route('/Trashbox_page', methods=['GET','POST'])
def affiche_trashbox():
	selected_u = session.get('my_user', None)
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]
	if os.path.isfile('trash.csv'):
		trash=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
			trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
			emails_to=df2[df2.To==selected_u]
			emails_From=df2[df2.From==selected_u]
			emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
			emails_to= emails_to[~emails_to.Indice_ancien.isin(trash_user.Indice_ancien)]
			emails_From= emails_From[~emails_From.Indice_ancien.isin(trash_user.Indice_ancien)]
			emails=trash_user[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
			len_dataset_trash=len(trash_user)
			len_dataset_greenbox=session.get('len_greenbox_estime', 0)

		else:
			df2=df.copy()
			emails_to=df2[df2.To==selected_u]
			emails_From=df2[df2.From==selected_u]
			emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
			column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
			emails = pd.DataFrame(columns = column_names)			
			len_dataset_trash=0
			len_dataset_greenbox=0
	else:
		df2=df.copy()
		emails_to=df2[df2.To==selected_u]
		emails_From=df2[df2.From==selected_u]
		emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
		column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
		emails = pd.DataFrame(columns = column_names)
		len_dataset_trash=0
		len_dataset_greenbox=0

	

	len_dataset_trash=len(emails)
	len_dataset_to=len(emails_to)
	len_dataset_from=len(emails_From)
	emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	if os.path.isfile('Greenbox.csv'):
		Greenbox_pd=pd.read_csv('Greenbox.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
			len_dataset_greenbox=session.get('len_greenbox_estime', 0)
		else:
			len_dataset_greenbox=0
	else:
		len_dataset_greenbox=0


	return render_template('Trashbox_page.html', nom=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

	

	

#Pagination previous button function Inbox
@app.route('/_affiche_prev', methods=['GET','POST'])
def affiche_prev():	
	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=len(trash_user)

			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=0
		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0

		len_dataset = session.get('len_dataset', 0)
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)

		
		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		emails=emails_to
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]

		#emails = df_test[df_test['To'].str.contains(selected_u) == True]
		nbre_mails = emails.shape[0]

		if  deb_nbre-20>0:
			deb_nbre = deb_nbre - 20
			fin_nbre = fin_nbre - 20
		else:
			deb_nbre = 0
			fin_nbre = 19
		emails = emails[deb_nbre:fin_nbre+1]	
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0

	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Inbox_page.html', nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


#Pagination next button function Inbox
@app.route('/_affiche_next', methods=['GET','POST'])
def affiche_next():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=len(trash_user)


			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=0

		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0

		len_dataset = session.get('len_dataset', 0)
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)

		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		emails=emails_to
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]

		#emails = df_test[df_test['To'].str.contains(selected_u) == True]
		nbre_mails = emails.shape[0]
		if deb_nbre+20<nbre_mails and fin_nbre+20<nbre_mails:
			deb_nbre = deb_nbre + 20
			fin_nbre = fin_nbre + 20
		elif deb_nbre+20<nbre_mails and fin_nbre+20>=nbre_mails:
			deb_nbre = fin_nbre
			fin_nbre = nbre_mails-1
		elif deb_nbre+20==nbre_mails:
			deb_nbre = deb_nbre
			fin_nbre = fin_nbre
		emails = emails[deb_nbre:fin_nbre+1]	
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0

	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Inbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


#Pagination previous button function sent box
@app.route('/_affiche_prev_sent', methods=['GET','POST'])
def affiche_prev_sent():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=len(trash_user)

			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=0

		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
			len_dataset_trash=0

		len_dataset = session.get('len_dataset', 0)
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)

		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		emails=emails_From
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]


		#emails = df_test[df_test['To'].str.contains(selected_u) == True]
		nbre_mails = emails.shape[0]

		if  deb_nbre-20>0:
			deb_nbre = deb_nbre - 20
			fin_nbre = fin_nbre - 20
		else:
			deb_nbre = 0
			fin_nbre = 19
		emails = emails[deb_nbre:fin_nbre+1]	
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0

	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Sentbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


#Pagination next button function sent
@app.route('/_affiche_next_sent', methods=['GET','POST'])
def affiche_next_sent():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=len(trash_user)


			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=0

		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			emails_From=emails_From[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
			len_dataset_trash=0

		len_dataset = session.get('len_dataset', 0)
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)

		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		emails=emails_From
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]


		#emails = df_test[df_test['To'].str.contains(selected_u) == True]
		nbre_mails = emails.shape[0]
		if deb_nbre+20<nbre_mails and fin_nbre+20<nbre_mails:
			deb_nbre = deb_nbre + 20
			fin_nbre = fin_nbre + 20
		elif deb_nbre+20<nbre_mails and fin_nbre+20>=nbre_mails:
			deb_nbre = fin_nbre
			fin_nbre = nbre_mails-1
		elif deb_nbre+20==nbre_mails:
			deb_nbre = deb_nbre
			fin_nbre = fin_nbre
		emails = emails[deb_nbre:fin_nbre+1]
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0
	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Sentbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

#Pagination previous button function greenbox
@app.route('/_affiche_prev_green', methods=['GET','POST'])
def affiche_prev_green():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		len_dataset = session.get('len_dataset', 0)
		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		#len_dataset_to = session.get('len_dataset_to', None)
		#len_dataset_from = session.get('len_dataset_from', None)
		
		
		### IIII 

		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				len_dataset_trash=len(trash_user)
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)

			else:
				df2=df.copy()
				len_dataset_trash=0
				len_dataset_greenbox=0
		else:
			df2=df.copy()
			len_dataset_trash=0
			len_dataset_greenbox=0


		emails_to=df2[df2.To==selected_u]
		emails_From=df2[df2.From==selected_u]
		emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
		#emails=pd.concat([emails_to,emails_From], axis=0, ignore_index=True)
		#print('Check len emails',len(emails))
		len_dataset_from=len(emails_From)
		len_dataset_to=len(emails_to)	


		emails=pd.read_csv('Greenbox.csv')
		emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien', 'importance', 'y_pred_from', 'y_pred_proba_from', 'y_pred_proba_to','y_pred_to','Pred_total','Indice_ancien']]
		len_dataset_greenbox = len(emails)
		#len_dataset_trash = session.get('len_dataset_trash', None)
		nbre_mails = emails.shape[0]
		if  deb_nbre-20>0:
			deb_nbre = deb_nbre - 20
			fin_nbre = fin_nbre - 20
		else:
			deb_nbre = 0
			fin_nbre = 19
		emails = emails[deb_nbre:fin_nbre+1]	

	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Greenbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

#Pagination next button function greenbox
@app.route('/_affiche_next_green', methods=['GET','POST'])
def affiche_next_green():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		#len_dataset = session.get('len_dataset', None)
		#len_dataset_to = session.get('len_dataset_to', None)
		#len_dataset_from = session.get('len_dataset_from', None)
		#len_dataset_trash = session.get('len_dataset_trash', None)

		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		#selected_u = request.form.get('selected_u')
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				len_dataset_trash=len(trash_user)
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)

			else:
				df2=df.copy()
				len_dataset_trash=0
				len_dataset_greenbox=0
		else:
			df2=df.copy()
			len_dataset_trash=0
			len_dataset_greenbox=0


		emails_to=df2[df2.To==selected_u]
		emails_From=df2[df2.From==selected_u]
		emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
		len_dataset_from=len(emails_From)
		len_dataset_to=len(emails_to)

		emails=pd.read_csv('Greenbox.csv')
		emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien', 'importance', 'y_pred_from', 'y_pred_proba_from', 'y_pred_proba_to','y_pred_to','Pred_total','Indice_ancien']]

		len_dataset_greenbox = len(emails)
		#len_dataset_trash = session.get('len_dataset_trash', None)

		nbre_mails = emails.shape[0]
		if deb_nbre+20<nbre_mails and fin_nbre+20<nbre_mails:
			deb_nbre = deb_nbre + 20
			fin_nbre = fin_nbre + 20
		elif deb_nbre+20<nbre_mails and fin_nbre+20>=nbre_mails:
			deb_nbre = fin_nbre
			fin_nbre = nbre_mails-1
		elif deb_nbre+20==nbre_mails:
			deb_nbre = deb_nbre
			fin_nbre = fin_nbre
		emails = emails[deb_nbre:fin_nbre+1]	
	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Greenbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

#Pagination previous button function trashbox
@app.route('/_affiche_prev_trash', methods=['GET','POST'])
def affiche_prev_trash():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
				emails_to= emails_to[~emails_to.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_From= emails_From[~emails_From.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails=trash_user[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=len(trash_user)

			else:
				df2=df.copy()
				emails_to=df2[df2.To==selected_u]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
				column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
				emails = pd.DataFrame(columns = column_names)			
				len_dataset_trash=0
		else:
			df2=df.copy()
			emails_to=df2[df2.To==selected_u]
			emails_From=df2[df2.From==selected_u]
			emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
			column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
			emails = pd.DataFrame(columns = column_names)
			len_dataset_trash=0

	


		len_dataset_trash=len(emails)
		len_dataset_to=len(emails_to)
		len_dataset_from=len(emails_From)
		emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]

		nbre_mails = emails.shape[0]

		if  deb_nbre-20>0:
			deb_nbre = deb_nbre - 20
			fin_nbre = fin_nbre - 20
		else:
			deb_nbre = 0
			fin_nbre = 19
		emails = emails[deb_nbre:fin_nbre+1]	
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0
	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Trashbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)


#Pagination next button function
@app.route('/_affiche_next_trash', methods=['GET','POST'])
def affiche_next_trash():	
	if request.method == 'POST':
		selected_u = session.get('my_user', None)
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		deb_nbre = int(request.form.get('aprevdeb'))
		fin_nbre = int(request.form.get('aprevfin'))
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)	
				emails_to= emails_to[~emails_to.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_From= emails_From[~emails_From.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails=trash_user[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]
				len_dataset_trash=len(trash_user)
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)

			else:
				df2=df.copy()
				emails_to=df2[df2.To==selected_u]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
				column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
				emails = pd.DataFrame(columns = column_names)			
				len_dataset_trash=0
				len_dataset_greenbox=0
		else:
			df2=df.copy()
			emails_to=df2[df2.To==selected_u]
			emails_From=df2[df2.From==selected_u]
			emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
			column_names = ['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']
			emails = pd.DataFrame(columns = column_names)
			len_dataset_trash=0
			len_dataset_greenbox=0

		


		len_dataset_trash=len(emails)
		len_dataset_to=len(emails_to)
		len_dataset_from=len(emails_From)
		emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien']]

		nbre_mails = emails.shape[0]
		if deb_nbre+20<nbre_mails and fin_nbre+20<nbre_mails:
			deb_nbre = deb_nbre + 20
			fin_nbre = fin_nbre + 20
		elif deb_nbre+20<nbre_mails and fin_nbre+20>=nbre_mails:
			deb_nbre = fin_nbre
			fin_nbre = nbre_mails-1
		elif deb_nbre+20==nbre_mails:
			deb_nbre = deb_nbre
			fin_nbre = fin_nbre
		emails = emails[deb_nbre:fin_nbre+1]	
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0
	else:
		print(json.dumps({'status':'Error not a post method'}))
	return render_template('Trashbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

@app.route("/statistics_box", methods=["GET", "POST"])
def statistics_box():
	if os.path.isfile('trash.csv'):
		selected_u = session.get('my_user', None)
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		trash_box=pd.read_csv('trash.csv')
		if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash_box.user.values:
			trash_box=trash_box[trash_box.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
			len_dataset_thread = session.get('len_dataset_thread', 0)
			len_dataset_supp_features = session.get('len_dataset_supp_features', 0)
			len_dataset_total_supp = session.get('len_dataset_total_supp', 0)
			len_dataset_trash = len(trash_box)
			df2 = df[~df.Indice_ancien.isin(trash_box.Indice_ancien)]			
			emails_to=df2[df2.To==selected_u]
			emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
			emails_From=df2[df2.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_to=len(emails_to)
			len_dataset_from=len(emails_From)
		#Greenbox 
		#Greenbox supprime
		#if len(trash_box)<len_dataset_total_supp:
		#	len_trash_sansthread_features=len(trash_box)
    	#else:
		#	len_trash_sansthread_features=len(trash_box)-len_dataset_total_supp
			if session.get('Est_passe_par_bonne_pratique')==False:#len(trash_box)<len_dataset_total_supp:
				len_trash_sansthread_features=len(trash_box)
				len_dataset_thread=0
				len_dataset_supp_features=0
			else:
				len_trash_sansthread_features=len(trash_box)-len_dataset_total_supp
				len_dataset_supp_features=len_dataset_total_supp-len_dataset_thread
			#GGGG
			len_dataset=session.get('len_dataset',0)
			if os.path.isfile('Greenbox.csv'):
				len_dataset_greenbox=session.get('len_greenbox_estime',0)
			else:
				len_dataset_greenbox=0

			taille_trash_co2 = 0
			now = datetime.datetime.now()

			for index, email in trash_box.iterrows():
				nbre_years = now.year-int(email.Date.split('-')[0])		
				if float(email['si_doc_ou_image']) != 0.0:
					taille_trash_co2 = taille_trash_co2 + 20 * nbre_years
				else:
					taille_trash_co2 = taille_trash_co2 + 10 * nbre_years

			data = {'Task' : 'Number of cleaned', 'Thread' : len_dataset_thread, 'Newletters, meetings, intern mails' : len_dataset_supp_features, 'Removal recommandation' : len_trash_sansthread_features}
			return render_template("Statistics3.html",nom=nom,mail=selected_u,data=data,users=liste_trois_user,len_dataset=len_dataset,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash,taille_trash_co2=taille_trash_co2)
		else:
			selected_u = session.get('my_user', None)
			nom=liste_trois_noms[liste_trois_user.index(selected_u)]
			
			len_dataset_trash = 0
			emails_to=df[df.To==selected_u]
			emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_to=len(emails_to)
			len_dataset_from=len(emails_From)
			#Greenbox 
			len_dataset_greenbox = 0
			#Greenbox supprime
			len_trash_sansthread_features=0
			len_dataset = session.get('len_dataset', 0)



			taille_trash_co2 = 0
			len_dataset_thread=0
			len_trash_sansthread_features=0
			len_dataset_supp_features=0
			data = {'Task' : 'Number of cleaned', 'Thread' : len_dataset_thread, 'Mots clés' : len_dataset_supp_features, 'Recommandation de suppression' : len_trash_sansthread_features}
			return render_template("Statistics3.html",nom=nom,mail=selected_u,data=data,users=liste_trois_user,len_dataset=len_dataset,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=0,len_dataset_trash=0,taille_trash_co2=0)

	else:
		selected_u = session.get('my_user', None)
		nom=liste_trois_noms[liste_trois_user.index(selected_u)]
		
		len_dataset_trash = 0
		emails_to=df[df.To==selected_u]
		emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
		emails_From=df[df.From==selected_u]
		emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
		len_dataset_to=len(emails_to)
		len_dataset_from=len(emails_From)
		#Greenbox 
		len_dataset_greenbox = 0
		#Greenbox supprime
		len_trash_sansthread_features=0
		len_dataset = session.get('len_dataset', 0)



		taille_trash_co2 = 0
		len_dataset_thread=0
		len_trash_sansthread_features=0
		len_dataset_supp_features=0
		data = {'Task' : 'Number of cleaned', 'Thread' : len_dataset_thread, 'Mots clés' : len_dataset_supp_features, 'Recommandation de suppression' : len_trash_sansthread_features}
		return render_template("Statistics3.html",nom=nom,mail=selected_u,data=data,users=liste_trois_user,len_dataset=len_dataset,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=0,len_dataset_trash=0,taille_trash_co2=0)


@app.route('/next', methods=['GET','POST'])
def affiche_mail(): 
	selected_u = session.get('my_user', None)
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]

	
	if request.method == "POST":
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]

				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
				emails_tot=pd.concat([emails_to,emails_From], axis=0, ignore_index=True)
				len_dataset_trash=len(trash_user)
				
				

			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=0
				len_dataset_greenbox=0
		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0
			len_dataset_greenbox=0
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)
		req = request.form['affiche_mail_seul']
		for v in request.form.values():
			x = ast.literal_eval(v)
			x=[n.strip() if type(n)==str else n for n in x]
		valeurs={'From': x[0], 'Mail': x[1], 'Date': x[2],'Subject':x[3],'Attached':int(x[4])}
		return render_template('Mail_seul2.html',adressemail=valeurs['From'],Mail=valeurs['Mail'],Date=valeurs['Date'],Subject=valeurs['Subject'],Attached=valeurs['Attached'],users=liste_trois_user,nom=nom,mail=selected_u,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

	else:
		print("detecte pas post")
		
	return render_template('Mail_seul2.html')

@app.route('/next_sent', methods=['GET','POST'])
def affiche_mail_sent(): 
	selected_u = session.get('my_user', None)
	nom=liste_trois_noms[liste_trois_user.index(selected_u)]

	
	if request.method == "POST":
		if os.path.isfile('trash.csv'):
			trash=pd.read_csv('trash.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash.user.values:
				trash_user=trash[trash.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]

				df2 = df[~df.Indice_ancien.isin(trash_user.Indice_ancien)]
				emails_to=df2[df2.To==selected_u]
				emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
				emails_From=df2[df2.From==selected_u]
				emails_From.drop_duplicates(subset=['Indice_ancien'], keep='first',inplace=True)
				emails_tot=pd.concat([emails_to,emails_From], axis=0, ignore_index=True)
				len_dataset_trash=len(trash_user)
				
				

			else:
				emails_to=df_affichage[df_affichage.To==selected_u]
				emails_From=df[df.From==selected_u]
				emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
				len_dataset_trash=0
				len_dataset_greenbox=0
		else:
			emails_to=df_affichage[df_affichage.To==selected_u]
			emails_From=df[df.From==selected_u]
			emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
			len_dataset_trash=0
			len_dataset_greenbox=0
		if os.path.isfile('Greenbox.csv'):
			Greenbox_pd=pd.read_csv('Greenbox.csv')
			if liste_trois_user_user[liste_trois_user.index(selected_u)] in Greenbox_pd.user.values:
				len_dataset_greenbox=session.get('len_greenbox_estime', 0)
			else:
				len_dataset_greenbox=0
		else:
			len_dataset_greenbox=0
		len_dataset_to = len(emails_to)
		len_dataset_from = len(emails_From)
		req = request.form['affiche_mail_sent_seul']
		for v in request.form.values():
			x = ast.literal_eval(v)
			x=[n.strip() if type(n)==str else n for n in x]
		valeurs={'From': x[0], 'Mail': x[1], 'Date': x[2],'Subject':x[3],'Attached':int(x[4])}
		return render_template('Mail_seul_sent.html',adressemail=valeurs['From'],Mail=valeurs['Mail'],Date=valeurs['Date'],Subject=valeurs['Subject'],Attached=valeurs['Attached'],users=liste_trois_user,nom=nom,mail=selected_u,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

	else:
		print("detecte pas post")
		
	return render_template('Mail_seul_sent.html')

@app.route('/Greenbox_page_clean', methods=['GET','POST'])
def clean_email():
	selected_u = session.get('my_user', None)
	session['Click_clean_mail']=True
	
	checked_mails_indexes = request.form.getlist('check')
	

	nom=liste_trois_noms[liste_trois_user.index(selected_u)]
	#Pour le From
	emails = pd.read_csv('Greenbox.csv')
	emails_todelete = []
	if os.path.isfile('trash.csv'):
		if len(checked_mails_indexes) != 0:
				emails_todelete = emails[emails['Indice_ancien'].isin(checked_mails_indexes)]
				emails_todelete=emails_todelete[['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','Cc']]
				emails_todelete.rename(columns={'si_doc_ou_image_y':'si_doc_ou_image'},inplace=True)
				emails_todelete.drop_duplicates(subset=['Indice_ancien'],inplace=True)
				emails_todelete['user']=liste_trois_user_user[liste_trois_user.index(selected_u)]
				for elem in emails_todelete.Indice_ancien.values:
					if emails_todelete[emails_todelete.Indice_ancien==elem].From.values==[selected_u]:
						if emails_todelete[emails_todelete.Indice_ancien==elem].To_ancien.str.contains(selected_u).values==True or emails_todelete[emails_todelete.Indice_ancien==elem].Cc.str.contains(selected_u).values==True:
							print('yesy')
							emails_todelete=emails_todelete.append(emails_todelete[emails_todelete.Indice_ancien==elem],ignore_index=True) 
				emails_todelete.to_csv('trash.csv', mode='a', header=False,index=False)
				emails = emails[~emails.Indice_ancien.isin(emails_todelete.Indice_ancien)]
				if os.path.exists('Greenbox.csv'):
					os.remove('Greenbox.csv')
				emails.to_csv('Greenbox.csv', index=False)

	else:
		if len(checked_mails_indexes) != 0:
				emails_todelete = emails[emails['Indice_ancien'].isin(checked_mails_indexes)]
				emails_todelete=emails_todelete[['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien','Cc']]
				emails_todelete.rename(columns={'si_doc_ou_image_y':'si_doc_ou_image'},inplace=True)
				emails_todelete.drop_duplicates(subset=['Indice_ancien'],inplace=True)
				emails_todelete['user']=liste_trois_user_user[liste_trois_user.index(selected_u)]
				for elem in emails_todelete.Indice_ancien.values:
					if emails_todelete[emails_todelete.Indice_ancien==elem].From.values==[selected_u]:
						if emails_todelete[emails_todelete.Indice_ancien==elem].To_ancien.str.contains(selected_u).values==True or emails_todelete[emails_todelete.Indice_ancien==elem].Cc.str.contains(selected_u).values==True:
							print('yesy')
							emails_todelete=emails_todelete.append(emails_todelete[emails_todelete.Indice_ancien==elem],ignore_index=True) 
				emails_todelete.to_csv('trash.csv',index=False)
				emails = emails[~emails.Indice_ancien.isin(emails_todelete.Indice_ancien)]
				if os.path.exists('Greenbox.csv'):
					os.remove('Greenbox.csv')
				emails.to_csv('Greenbox.csv', index=False)

	emails=emails[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image_y','From_ancien','To_ancien', 'importance', 'y_pred_from', 'y_pred_proba_from', 'y_pred_proba_to','y_pred_to','Pred_total','Indice_ancien']]
	trash_box=pd.read_csv('trash.csv',header=0)
	if liste_trois_user_user[liste_trois_user.index(selected_u)] in trash_box.user.values:
		trash_user=trash_box[trash_box.user==liste_trois_user_user[liste_trois_user.index(selected_u)]]
		len_dataset_trash = len(trash_user)
	else:
		len_dataset_trash=0
	df2 = df[~df.Indice_ancien.isin(trash_box.Indice_ancien)]			
	emails_to=df2[df2.To==selected_u]
	emails_to=emails_to[['From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','From_ancien','To_ancien','To']]
	emails_From=df2[df2.From==selected_u]
	emails_From=emails_From.groupby(['Indice_ancien','From','Mail_not_clean','Date','Subject_not_clean','si_doc_ou_image','user','From_ancien','To_ancien'])['si_question'].mean().reset_index()
	len_dataset_to=len(emails_to)
	len_dataset_from=len(emails_From)

	
	len_dataset_greenbox=session.get('len_greenbox_estime')-len(emails_todelete)
	session['len_greenbox_estime']=session.get('len_greenbox_estime')-len(emails_todelete)
	session['len_dataset_trash'] = session.get('len_dataset_trash', 0)+len(emails_todelete)
	nbre_mails = emails.shape[0]
	deb_nbre = 0
	fin_nbre = 19
	emails = emails[deb_nbre:fin_nbre+1]
	
	return render_template('Greenbox_page.html',nom=nom,mail=selected_u,users_dropdown = liste_trois_user, tables=emails.iterrows(), selected_u=selected_u, nbre_mails = nbre_mails, deb_nbre= deb_nbre, fin_nbre = fin_nbre,len_dataset_to=len_dataset_to,len_dataset_from=len_dataset_from,len_dataset_greenbox=len_dataset_greenbox,len_dataset_trash=len_dataset_trash)

#catching 404 error
@app.errorhandler(404) 
def not_found(e): 
	 return redirect(url_for('home'))

#catching 405 error
@app.errorhandler(405) 
def not_founds(e): 
	return redirect(url_for('home'))

if __name__ == '__main__':
        app.run(debug=True, host='127.0.0.1', port=8053)
