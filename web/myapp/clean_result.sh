cd /home/ubuntu/catbridge/
sudo su
chmod 777 myapp

find /home/ubuntu/catbridge/myapp/result -maxdepth 1 -type d -mtime +1 -exec rm -r {} \;
find /home/ubuntu/catbridge/myapp/static -maxdepth 1 -type d -name '169*' -mtime +1 -exec rm -r {} \;
