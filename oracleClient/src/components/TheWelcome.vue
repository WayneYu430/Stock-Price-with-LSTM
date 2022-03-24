<script>
import axios from 'axios'

const API_URL = 'http://127.0.0.1:5000/'
const API_URL_GET = 'http://127.0.0.1:5000/'
const API_URL_POST = 'http://127.0.0.1:5000/question'
const API_URL_POST_IMAGE = 'http://127.0.0.1:5000/image'
export default {

  data() {
    return {
      text: 'What is the prediction data for AAPL in 2021-9-20 ?',
      message: 'Edit Me',
      postRes: null,
      image:null
    }
  },
  methods: {
    pushButton() {
      // this.message = this.message.split('').reverse().join('')
      // this.fetchData()
      // this.testAxiosGet()
      const comList = "list"
      const comName = "name"
      const comHist = "history"
      const comPredict = "prediction"
      const comData = "data"
      if(this.text.includes(comList)){
        this.askComList()
      }else if(this.text.includes(comName)){
        this.askComName()
      }else if(this.text.includes(comHist)){
        this.askComHist()
      }else if(this.text.includes(comPredict) && this.text.includes(comData)){
        this.askComPredictData()
      }else if(this.text.includes(comPredict)){
        this.askComPredict()
      }else{
        this.errorMsg()
      }

    },
    fetchData() {
      const url = `${API_URL_GET}`
      fetch(url).then(res=>res.json()).then(console.log(this.res)).then(data => (this.res = data.Date))
      
    },
    notify() {
      alert('navigation was prevented.')
    },
    askComList(){
      let url_tmp = API_URL+'comList'
      axios.get(url_tmp).then(res => {
        this.postRes = res.data
      })
    },
    askComName(){
      let formData = new FormData()
      let url_tmp = API_URL+'comName'
      let array_tmp = this.text.split(" ")
      const com_symbol = array_tmp[array_tmp.length-2]
      formData.append('com_symbol', com_symbol);
      axios.post(url_tmp,formData).then(res => {
        this.postRes = res.data
      })
    },
    askComHist(){
      let formData = new FormData()
      let url_tmp = API_URL+'comName'
      let array_tmp = this.text.split(" ")
      const com_symbol = array_tmp[array_tmp.length-2]

      formData.append('com_symbol', com_symbol);
      axios.post(url_tmp,formData).then(res => {
        this.postRes = res.data
      })
    },
    askComHist(){
      let formData = new FormData()
      let url_tmp = API_URL+'comHist'
      let array_tmp = this.text.split(" ")
      const com_symbol = array_tmp[array_tmp.length-2]
      formData.append('com_symbol', com_symbol);
      axios.post(url_tmp,formData,       
      {
         headers:{'Content-type': 'image/png'},
         responseType: 'blob'
       }).then(res => {
        const urlCreator = window.URL || window.webkitURL
        this.image = urlCreator.createObjectURL(res.data)

      })
    },
    askComPredict(){
      let formData = new FormData()
      let url_tmp = API_URL+'PredictImg'
      let array_tmp = this.text.split(" ")
      const com_symbol = array_tmp[array_tmp.length-4]
      const ask_date = array_tmp[array_tmp.length-2]
      formData.append('com_symbol', com_symbol);
      formData.append('ask_date', ask_date);
      axios.post(url_tmp,formData,       
      {
         headers:{'Content-type': 'image/png'},
         responseType: 'blob'
       }).then(res => {
        const urlCreator = window.URL || window.webkitURL
        this.image = urlCreator.createObjectURL(res.data)

      })
    },
    askComPredictData(){
      let formData = new FormData()
      let url_tmp = API_URL+'PredictData'
      let array_tmp = this.text.split(" ")
      const com_symbol = array_tmp[array_tmp.length-4]
      const ask_date = array_tmp[array_tmp.length-2]
      formData.append('com_symbol', com_symbol);
      formData.append('ask_date', ask_date);
      axios.post(url_tmp,formData).then(res => {
        this.postRes = res.data
      })
    },
    testAxiosGet(){
      axios.get(API_URL_GET).then(res => {
        this.res = res.data
      })
    },
    testAxiosPost(){
      let formData = new FormData()
      formData.append('name', 'AAPL');
      formData.append('date', '2021-9-14')
      axios.post(API_URL_POST_IMAGE, formData, 
       {
         headers:{'Content-type': 'image/png'},
         responseType: 'blob'
       }
       ).then(res=>{
          console.log(res)
          const urlCreator = window.URL || window.webkitURL
          this.image = urlCreator.createObjectURL(res.data)

      })
    },
    errorMsg(){
      axios.get(API_URL_GET+'error').then(res => {
        this.postRes = res.data
      })
    },
  },
  mounted(){
    
  }
}
</script>

<template>
  <h2>Ask for Questions</h2>
  <h3>Some sample Questions:</h3>
  <p>- What is stock [list] ?</p>
  <p>- What is the name for [AAPL] ?</p>
  <p>- What is the [history] image for [AAPL] ?</p>
  <p>- What is the [prediction] data for [AAPL] in [2021-9-20] ?</p>
  <p>- What is the [prediction] image for [AAPL] in [2021-9-20] ?</p>
  <div>
    <input v-model="text" style="width:400px;height=100px">
     <button @click="pushButton" style="width:80px;height=100px">
    {{ message }}
  </button>
  </div>
  <h3>The Answer is: </h3>
  <p>{{ postRes }} </p>
  <div>
  <img :src="image"   style="width:100%;height=100%" >
  </div>
</template>