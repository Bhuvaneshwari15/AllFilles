const http=require('http');
const url=require('url');
const fs=require('fs');
const path=require('path');
const PORT=1800;
const mimeType=
{
'.js':'text/javascript',
'.html':'text/html',
'.xml':'text/xml',
'.jpg': 'image/jpeg'
};
http.createServer((req,res)=>{
const parsedUrl=url.parse(req.url);
if(parsedUrl.pathname=== "/")
{
var filesLink="<ul>";
res.setHeader('content-type','text/html');
var filesList=fs.readdirSync("./");
filesList.forEach(element=>
{
if(fs.statSync("./"+element).isFile())
{
filesLink+=`<br/><li><a href='./${element}'>${element}</a></li>`;
}
});
filesLink+="</ul>";
res.end("<h1>List of files:</h1>"+filesLink);
}
const sanitizepath=
path.normalize(parsedUrl.pathname).replace(/^(\.\.[\/\\])+/,' ');
let pathname=path.join(__dirname,sanitizepath);
if(!fs.existsSync(pathname))
{
res.statusCode=404;
res.end(`File ${pathname}not found!`);
}
else
{
fs.readFile(pathname, function(err,data)
{
if(err)
{
res.statusCode=500;
res.end(`Error in getting the file.`); }
else
{
const ext=path.parse(pathname).ext;
res.setHeader('content-Type',mimeType[ext] || 'text/plain');
res.end(data);
}
}
);
}
}).listen(PORT);