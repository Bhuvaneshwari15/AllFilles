var express=require("express")
var app=express();
var path=require("path");
var mysql=require("mysql");
var bodyparser=require("body-parser");
app.use(bodyparser.urlencoded({extended:false}));
app.use(bodyparser.json());
var con=mysql.createConnection({
host:"localhost",
user:"root",
password:"admin",
database:"full"
});
app.get('/',function(req,res){
res.sendFile(path.join(__dirname+'/index.html'));
});
app.post('/submit',function(req,res){
var name=req.body.name;
var email=req.body.email;
var username=req.body.username;
let alert=require('alert');
alert("record inserted");
con.connect(function(err){
if(err)throw err;
var sql="INSERT INTO forms(name,email,username)values('"+name+"','"+email+"','"+username+"')";
con.query(sql,function(err,result){
if(err)throw err;
console.log("1 record inserted");
res.end();
});
});
})
app.listen(3000);
console.log("running at port 3000");