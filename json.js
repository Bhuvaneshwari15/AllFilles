var express=require('express');
var app=express();
var fs=require('fs');
app.get('/index.html',function(req,res) {
res.sendFile(__dirname+"/" + "index.html");
})
app.get('/process_get',function(req,res,next) {
response={first_name:req.query.first_name,last_name:req.query.last_name};
console.log(response);
const name=JSON.stringify(response);
fs.writeFileSync('data.json',name);
res.redirect('/user')
app.get('/user',function(req,res,next) {
res.send(JSON.stringify(response));
});
res.end();
})
var server=app.listen(8000,function() {
var host=server.address().address
var port=server.address().port
console.log("Example app listening at http://%s:%s",host,port)
})