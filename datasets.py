import cv2
import numpy as np
import sqlite3


faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);

def insertorupdate(Id,Name,Dob):
    conn=sqlite3.connect("sqlite.db")
    cmd="SELECT * FROM STUDENTS WHERE ID="+str(Id);
    cursor=conn.execute(cmd);
    isRecordExist=0;
    for row in cursor:
        isRecordExist=1;
    if(isRecordExist==1):
        conn.execute("UPDATE students SET Name=? WHERE Id=?",(Name,Id,))
        conn.execute("UPDATE students SET Dob=? WHERE Id=?", (Dob, Id,))
    else:
        conn.execute("INSERT INTO STUDENTS(Id,Name,Dob) values(?,?,?)",(Id,Name,Dob))

    conn.commit()
    conn.close()

#insert user defined values into table

Id=input('Enter User VoterID:')
Name=input('Enter User Name: ')
Dob=input('Enter User Dob in ddmmyy:')

insertorupdate(Id,Name,Dob)

#dataset face

sampleNum=0;
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataset/user."+str(Id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(100)
    cv2.imshow("Face",img)
    cv2.waitKey(1);
    if(sampleNum>20):
        break;
cam.release()
cv2.destroyAllWindows()