                                                                                           
                                                                                        
                                                                                             
 myLoadUrl = async function(){
  //alert('The test function will need to be changed if other models are loaded')                                                                                             
  const myFileName = document.getElementById('myInFile').value
  if (myFileName != null){  
    model = await tf.loadModel(myFileName);     // should make the model a global variable
   // await myPredict()
  }                                                                           
}                                                                                             
                                                                                             
                                                                                             

myPredict = async function(){
   // document.getElementById('myButtonTest').style.backgroundColor = 'red'
  //  console.log('Model Predict')
    
                                                                                             
  const image = tf.fromPixels(document.getElementById('my32x32CanvasA')).toFloat().sub(tf.scalar(127.5)).div(tf.scalar(127.5)).reshape([1, 224, 224, 3]) ;
                                                                                             
                                                                                         
                                                                                             
   //   console.log('frompixels')                                                                                           
  const inferenceResult = await model.predict(image);
  //console.log('await model.predict(image);')   
                                                                                             
  const {values, indices} = await tf.topk(inferenceResult, 5);     // grab the top 7 outputs


//values.print();
//indices.print();                                                                                           
  myValues = values.dataSync()                                                                              
  myIndices = indices.dataSync()                                                                                           
                                                                                             
                                                                                         
  document.getElementById('myDivTest').innerHTML = ''   // clear the div    
  for (let x=0; x< myIndices.length; x++){                                                                                           
      document.getElementById('myDivTest').innerHTML += myIndices[x]+', <b>'+Math.round(myValues[x]*100)+'%</b>: '+IMAGENET_CLASSES[myIndices[x]] + '<br>'                                                                                         
   }                                                                                          

}         