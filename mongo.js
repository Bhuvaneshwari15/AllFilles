const {MongoClient}=require('mongodb');
const url='mongodb://localhost:27017';
const client=new MongoClient(url);
const dbName='bhuvi';
async function main(){
await client.connect();
console.log('connected successfully to server');
const db=client.db(dbName);
const collection=db.collection('MCA');
const insertResult=await collection.insertMany([{name:'raj',dept:'mca'},{name:'arun',dept:'mca'},{name:'naveen',dept:'mca'},{name:'praveen',dept:'mca'}]);
console.log('inserted documents=>',insertResult);
const findResult=await collection.find({}).toArray();
console.log('found documents=>',findResult);
const filteredDocs=await collection.find({name:naveen}).toArray();
console.log('found documents filtered by {name:naveen}=>',filteredDocs);
const updateResult=await collection.updateOne({name:'raj'},{$set:{name:'raja'}});
console.log('updated documents=>',updateResult);
const deleteResult=await collection.deleteMany({name:'praveen'});
console.log('deleted documents=>',deleteResult);
const find1=await collection.find({}).toArray();
console.log('found documents=>',find1);
const drop=await collection.drop();
console.log('dropped documents=>',drop);
const find2=await collection.find({}).toArray();
console.log('found documents=>',find2);
return 'done';
}
main()
.then(console.log)
.catch(console.error)
.finally(()=>client.close());