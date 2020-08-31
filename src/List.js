import React, { Component } from 'react'
import  { useState, useEffect } from 'react';
import { getList, addToList, deleteItem, updateItem } from './ListFunctions'
import Button from './Button';
import Select from './Select';
import {Line} from 'react-chartjs-2';
import Menu from './Menu';
import Switcho from './switcho';
import Menu2 from './Menu2';
import { makeStyles } from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import Switch from '@material-ui/core/Switch';
import axios from 'axios'
export default class List extends React.Component {
    constructor() {
        super()
        this.state = {
            id: '',
            term: '',
            editDisabled: false,
            items: [],
            persons: [],
            p: "",
            Data2: true,
            Data1: 'Yes',
            Training:"",
            Test:""
        }

        this.onSubmit = this.onSubmit.bind(this)
        this.onChange = this.onChange.bind(this)
    }

    componentDidMount () {
        this.getAll()
    }

    onChange = e => {
        this.setState({
            term: e.target.value,
            editDisabled: 'disabled'
        })
    }
    getListt = () => {
        return axios
            .get('api/tasks', {
                headers: { "Content-type": "application/json" }
            })
            .then(res => {
                this.setState({
                    term: res.title,
                   
                }) ;
                
                
                
            })}
    
    getAll = () => {
        getList().then(data => {
            this.setState({
                term: data,
                items: [...data]
            },
                () => {
                    
                    console.log(this.state.term)
                })
        })
    }
     addToList = term => {
        return axios
            .post(
                'api/task', {
                    title: term
                }, {
                    headers: { "Content-type": "application/json" }
                })
            .then((res) => {
                this.setState({
                    term: res.Aziz
                   
                }) ;
            })
    }
    componentDidMount = async () =>{
     
       axios.get(`api/task`)
       .then(res => {
         
         this.setState({ term:res.a });
       })
     }
    onSubmit = e => {
        e.preventDefault()
        this.setState({ editDisabled: '' })
        addToList(this.state.term).then(() => {
            this.getAll()
        })
    }

    onUpdate = e => {
        e.preventDefault()
        updateItem(this.state.term, this.state.id).then(() => {
            this.getAll()
        })
    }

    onEdit = (item, itemid, e) => {
        e.preventDefault()
        this.setState({
            id: itemid,
            term: item
        })
        console.log(itemid)
    }


     getSVM() {
        axios.get(`http://127.0.0.1:5000/api/svm`)
          .then(res => {
            
            this.setState({Training:res.data });
          })
          axios.get(`http://127.0.0.1:5000/api/svm2`)
          .then(res => {
            
            this.setState({Test:res.data });
          })}
          getKN() {
            axios.get(`http://127.0.0.1:5000/api/Neighbors`)
              .then(res => {
                
                this.setState({Training:res.data });
              })
              axios.get(`http://127.0.0.1:5000/api/Neighbors2`)
              .then(res => {
                
                this.setState({Test:res.data });
              })}
          getbayes() {
            axios.get(`http://127.0.0.1:5000/api/bayes`)
              .then(res => {
                
                this.setState({Training:res.data });
              })
              axios.get(`http://127.0.0.1:5000/api/bayes2`)
              .then(res => {
                
                this.setState({Test:res.data });
              })}
              getarbre() {
                axios.get(`http://127.0.0.1:5000/api/arbre`)
                  .then(res => {
                    
                    this.setState({Training:res.data });
                  })
                  axios.get(`http://127.0.0.1:5000/api/arbre2`)
                  .then(res => {
                    
                    this.setState({Test:res.data });
                  })}
changedata(){
    this.setState({Data1:false });
    console.log(this.state.Data1);

}
                    
    onDelete = (val, e) => {
        e.preventDefault()
        deleteItem(val)

        var data = [...this.state.items]
        data.filter((item, index) => {
            if (item[1] === val) {
                data.splice(index, 1)
            }
            return true
        })
        this.setState({ items: [...data] })
    }

    render () {
       
        const Subm1 = {
            color: "white",
            'borderRadius':'15px',
            backgroundColor: "HotPink",
            padding: "10px",
            fontFamily: "Arial",
            width:"280px",
            marginLeft: '15rem' 
          };
          const Subm2 = {
            color: "white",
            borderWidth:"2px",
            borderRadius:'15px',
            backgroundColor: "Fuchsia",
            padding: "10px",
            fontFamily: "Arial",
            width:"280px",
            marginLeft: '15rem' 
          };
          const Subm3 = {
            color: "white",
            'borderRadius':'15px',
            backgroundColor: "DodgerBlue",
            padding: "10px",
            fontFamily: "Arial",
            width:"280px",
            height:"40px",
            marginLeft: '15rem' 
          };
        const Subm4 = {
            color: "white",
            'borderRadius':'15px',
            backgroundColor: "BlueViolet",
            padding: "10px",
            fontFamily: "Arial",
            width:"280px",
            marginLeft: '15rem' 
          };
          const text = {
            color: "white",
           
            padding: "10px",
            fontFamily: "Arial",
            width:"300px",
            Bottom:"-400px",
            height:"200px",
            left:"500px"
          };
          
          const data = {
            labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
            datasets: [
              {
                label: 'My First dataset',
                fill: false,
                lineTension: 0.1,
                backgroundColor: 'rgba(75,192,192,0.4)',
                borderColor: 'rgba(75,192,192,1)',
                borderCapStyle: 'butt',
                borderDash: [],
                borderDashOffset: 0.0,
                borderJoinStyle: 'miter',
                pointBorderColor: 'rgba(75,192,192,1)',
                pointBackgroundColor: '#fff',
                pointBorderWidth: 1,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: 'rgba(75,192,192,1)',
                pointHoverBorderColor: 'rgba(220,220,220,1)',
                pointHoverBorderWidth: 2,
                pointRadius: 1,
                pointHitRadius: 10,
                data: [65, 59, 80, 81, 56, 55, 40]
              }
            ]
          };
          const mystyle = {
            color: "white",
            'borderRadius':'10px',
            backgroundColor: "Purple",
            padding: "10px",
            fontFamily: "Arial",
            bottom:"100px",
            width:"600px",
            height:"60px",
            left:"300px"
          };
        return (
            <div className="col-md-12">
                <Menu2/> 
               
              <Switcho  />
             
                <ul>
               
      </ul>
               
                <form noValidate autoComplete="off" onSubmit={this.onSubmit}>
                <div>  <h4 style={mystyle} > {this.state.Training}</h4>  <h4 style={mystyle} >  {this.state.Test}</h4></div>                 
                    <button className="btn btn-success btn-block"
                        type="submit"
                        onClick={this.getSVM.bind(this)} style={Subm3}>
                        Classification Support Vector Machine
                    </button>
                    <button className="btn btn-success btn-block"
                        type="submit"
                        onClick={this.getbayes.bind(this)} style={Subm3}>
                        Classification Gaussian Naive Bayes
                    </button>
                    <button className="btn btn-success btn-block"
                        type="submit"
                        onClick={this.getKN.bind(this)} style={Subm3}>
                        Classification K Nearest Neighbors
                    </button>
                    <button className="btn btn-success btn-block"
                        type="submit"
                        onClick={this.getarbre.bind(this)} style={Subm3}>
                        <h6>Classification Decision Tree</h6>
                    </button>
                    
         
   </form>
                
               
            </div>
        )
    }
}
