library(shiny)
library(readr)
library(rpart)
library(ggplot2)
library(naivebayes)
library(e1071)
library(fpc)
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulationinsta
library(gtools) # for discretisation
library(corrplot)
library(Hmisc)
library(devtools)
library(PerformanceAnalytics)
library(FactoMineR)
library(arules)
library(pdist)
library(class)
library(arulesViz)
library(methods)
library(DT)
library(ISLR)
library(caret)
require(tree)
library(labelled)
library(klaR)
library(questionr)
library(grDevices)
library(shinythemes)

creditcard <- read_csv("creditcard.csv")

#check for NULL values 
anyNA(creditcard)

#change anomalous values
for(i in 1:nrow(creditcard)){
  for(j in 7:12){
    if (creditcard[i,j]==0 |creditcard[i,j]==-2) 
      creditcard[i,j]= -1
  }
}


knn_data <- creditcard 

numeric_dataset <- creditcard

numeric_dataset$dpnm <- factor(numeric_dataset$dpnm, levels = c(0,1), labels = c("NO", "YES"))
#numeric_dataset = subset(numeric_dataset, select = -c(ID))




#Creating New Variable
creditcard$workstate <- ""

for (i in 1:nrow(creditcard)) {
  if ((creditcard[i,7] + creditcard[i,8] +creditcard[i,9]+creditcard[i,10] +creditcard[i,11]+creditcard[i,12]) <= 0){
    creditcard[i,26] <- "YES"  
  }
  else {
    creditcard[i,26] <- "NO"         
  }
}
creditcard$workstate<- factor(creditcard$workstate)


#remove variables
creditcard = subset(creditcard, select = -c(ID,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6))


#convert 
creditcard$dpnm <- factor(creditcard$dpnm, levels = c(0,1), labels = c("NO", "YES"))
creditcard$SEX <- factor(creditcard$SEX, levels = c(1,2), labels = c("MALE", "FEMALE"))
creditcard$MARRIAGE <- factor(creditcard$MARRIAGE, levels = c(1,2,3), labels = c("married", "single", "others"))
creditcard$EDUCATION <- factor(creditcard$EDUCATION, levels = c(1,2,3,4,5,6), labels = c("graduate school", "university", "high school", "others", "unknown", "unknown"))

creditcard$AGE<-cut(creditcard$AGE,c(20,30,40,50,60,70))
creditcard$AGE <- factor(creditcard$AGE)




##############################APRIORI##############################

apriori_dataset<- creditcard
apriori_dataset = subset(apriori_dataset, select = -c(BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6))
apriori_dataset$LIMIT_BAL <- discretize(apriori_dataset$LIMIT_BAL, method = "interval", breaks = 5, 
                                        labels = c("lower","low", "medium", "high", "higher"), order = T)



##############################  TRAIN AND TEST ##############################

# Shuffle the dataset; build train and test
n <- nrow(numeric_dataset)
shuffled <- numeric_dataset[sample(n),]
train <- shuffled[1:round(0.7 * n),]
test <- shuffled[(round(0.7 * n) + 1):n,]



##############################  DECISION TREE##############################

train_dt=sample(1:nrow(creditcard), (0.7 * n))

tree.creditcard = tree(dpnm~., data=creditcard)


#Decision tree with origical dataset
plot(tree.creditcard)
text(tree.creditcard, pretty = 0)
tree.creditcard

#Decision tree with train_dt dataset

set.seed(101)
tree.creditcard = tree(dpnm~., creditcard, subset=train_dt)
plot(tree.creditcard)
text(tree.creditcard, pretty=0)


tree.pred1 = predict(tree.creditcard, creditcard[-train_dt,], type="class")
with(creditcard[-train_dt,], table(tree.pred1, dpnm))
table_acc<- (with(creditcard[-train_dt,], table(tree.pred1, dpnm)))

#With cross Validation

#cv.creditcard = cv.tree(tree.creditcard, FUN = prune.misclass)
#cv.creditcard


#downward spiral part because of the misclassification error on cross-validated points
#plot(cv.creditcard)


#best=the max size of the tree
prune.creditcard = prune.misclass(tree.creditcard, best =2)
plot(prune.creditcard)
text(prune.creditcard, pretty=0)

tree.pred2 = predict(prune.creditcard, creditcard[-train_dt,], type="class")
with(creditcard[-train_dt,], table(tree.pred2, dpnm))

table_acc2<- (with(creditcard[-train_dt,], table(tree.pred2, dpnm)))


############################## KNN ##############################

knn_data = subset(knn_data, select = -c(ID,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6))

#Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }

knn_data.subset.n <- as.data.frame(lapply(knn_data[,2:18], normalize))



set.seed(123)
dat.d <- sample(1:nrow(knn_data.subset.n),size=nrow(knn_data.subset.n)*0.7,replace = FALSE) #random selection of 70% data.

traink <- knn_data[dat.d,] # 70% training data
testk <- knn_data[-dat.d,] # remaining 30% test data

#Creating seperate dataframe for 'dpnm' feature which is our target.
traindpnm <- knn_data[dat.d,18,drop = TRUE ]
testdpnm <-knn_data[-dat.d,18,drop = TRUE]


i=1
k.optm=1
for (i in 1:20){
  knn.mod <- knn(train=traink, test=testk, cl=traindpnm, k=i)
  k.optm[i] <- 100 * sum(testdpnm == knn.mod)/NROW(testdpnm)
  k=i
  cat(k,'=',k.optm[i],'')
}


##############################NAIVE BAYES##############################

nvData<-creditcard


nvData$LIMIT_BAL <- discretize(nvData$LIMIT_BAL, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT1 <- discretize(nvData$BILL_AMT1, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT2 <- discretize(nvData$BILL_AMT2, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT3 <- discretize(nvData$BILL_AMT3, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT4 <- discretize(nvData$BILL_AMT4, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT5 <- discretize(nvData$BILL_AMT5, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$BILL_AMT6 <- discretize(nvData$BILL_AMT6, method = "interval", breaks = 5, 
                               labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT1 <- discretize(nvData$PAY_AMT1, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT2 <- discretize(nvData$PAY_AMT2, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT3 <- discretize(nvData$PAY_AMT3, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT4 <- discretize(nvData$PAY_AMT4, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT5 <- discretize(nvData$PAY_AMT5, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)
nvData$PAY_AMT6 <- discretize(nvData$PAY_AMT6, method = "interval", breaks = 5, 
                              labels = c("lower","low", "medium", "high", "higher"), order = T)


#split the creditcard dataset into training and test dataset
indxTrain <- createDataPartition(y = nvData$dpnm,p = 0.75,list = FALSE)
training <- nvData[indxTrain,] 
testing <- nvData[-indxTrain,]


#Check dimensions of the split
prop.table(table(nvData$dpnm)) * 100
prop.table(table(training$dpnm)) * 100
prop.table(table(testing$dpnm)) * 100

x = nvData[,-18]
y = nvData$dpnm



############################################################



#--------------------------------------------------- UI CODE --------------------------------------------------------------

shinyApp(  
  ui = tagList(
    navbarPage( "flatly",
                theme = shinytheme("flatly"),
                
                #####################################  DATASET'S INFO ##########################################   
                tabPanel("Creditcard Dataset", 
                         pageWithSidebar(
                           h4('Creditcard Dataset'),
                           sidebarPanel(
                             numericInput(inputId = "obs",
                                          label = "Number of observations to view:",
                                          value = 10),
                             helpText("Note: while the data view will show only the specified",
                                      "number of observations, the summary will still be based",
                                      "on the full dataset."),
                             width=2
                             
                           ),
                           mainPanel(
                             tabsetPanel(type = "tabs",
                                         
                                         tabPanel("Summary", verbatimTextOutput("summary0"), tableOutput("table0") ),
                                         
                                         tabPanel("summary After Preprocessing ", verbatimTextOutput("summaryAfter"), tableOutput("tableAfter")),
                                         
                                         # Output: Formatted text for caption ----
                                         h3(textOutput("caption", container = span)),
                                         
                                         # Output: Verbatim text for data summary ----
                                         verbatimTextOutput("summary"),
                                         
                                         # Output: HTML table with requested number of observations ----
                                         tableOutput("view")
                                         
                                         
                                         
                             )
                             
                           ))
                         
                ),
                ###################################CLASSIFICATION ############################################    
                
                tabPanel("Classification", 
                         
                         mainPanel(
                           tabsetPanel(
                             tabPanel("Desicion Tree",
                                      sidebarPanel(radioButtons('prun', 'Apply Prunning:',
                                                                c(No = 'No_Prunning',
                                                                  Yes = 'Prunning')),
                                                   width = 3
                                      ),
                                      mainPanel( verticalLayout(
                                        plotOutput("treeplot"),
                                        
                                        verbatimTextOutput("accur")
                                      )
                                      )
                             ),
                             
                             tabPanel("Naive Bayes", 
                                      
                                      mainPanel(
                                        verbatimTextOutput("NBayes") 
                                      ),
                                      width= 3
                                      
                             ),
                             
                             
                             tabPanel("KNN ", 
                                      sidebarPanel(
                                        numericInput(inputId = "kn",
                                                     label = "Number of k-Nearest Neighbors: ",
                                                     value = 10),
                                        width = 3
                                      ),
                                      mainPanel(
                                        plotOutput("plotKNN"),verbatimTextOutput("knn1")
                                        
                                      ))
                             
                             
                           )
                         )),
                ########################################VISUALIZATION/PLOTS############################################     
                
                tabPanel("Visualization",
                         pageWithSidebar(
                           h4('Visualize the dataset'),
                           sidebarPanel(
                             
                             selectInput('xcol', 'X Variable', names(creditcard)),
                             selectInput('ycol', 'Y Variable', names(creditcard),
                                         selected=names(creditcard)[[2]]),
                             
                             width = 3
                           ),
                           mainPanel(
                             tabsetPanel(
                               tabPanel("Histogramm", tags$br(), 
                                        plotOutput("plot1"), downloadButton('download_hist', 'Download Plot', class = "buttCol")),
                               tabPanel("Barplot", tags$br(), 
                                        plotOutput("plot2"),downloadButton('download_BarPlot', 'Download Plot', class = "buttCol")),
                               tabPanel("Scatter Plot",tags$br(), 
                                        plotOutput("plot3"),downloadButton('download_ScatterPlot', 'Download Plot', class = "buttCol"))
                               
                               
                             )
                             
                           ))
                ),
                #############################################ASSOCIATION RULES###################################################     
                
                tabPanel("Association Rules", 
                         sidebarPanel(
                           
                           h4( strong( "Choose parameters for Apriori Algorithm" )),
                           
                           tags$br(),
                           
                           radioButtons('rhs', 'Rhs has only class variable(dpnm):',c(No = 'No_rhsdpnm',
                                                                                      Yes = 'rhsdpnm')),
                           
                           sliderInput('sup', "Support", min = 0.001, max = 1, value = 0.3, step = 0.05),
                           
                           sliderInput('conf', 'Confidence', min = 0.01, max =1, value = 0.5, step = 0.05),
                           
                           sliderInput('len', 'Minimum Rule Length', min = 1, max =7, value = 1, step = 1),
                           
                           sliderInput('mlen', 'Maximum Rule Length', min = 1, max =7, value = 4, step = 1),
                           
                           sliderInput('time', 'Maximum Time Taken', min = 1, max =25, value = 3, step = 1),
                           
                           
                           downloadButton('downloadData', 'Download Rules as CSV'),
                           width = 3
                           
                         ),
                         
                         
                         mainPanel(
                           tabsetPanel(
                             tabPanel('Rules', value = 'datatable',  DT::dataTableOutput("rules")),
                             
                             tabPanel('Plot',  plotOutput("scatterPlot")),
                             tabPanel('Frequency Plot',  plotOutput("FreqPlot")),
                             tabPanel('Graph Plot',  plotOutput("graphPlot")),
                             tabPanel('groupedMatrix Plot',  plotOutput("groupedMatrix"))
                             
                             
                           )
                           
                         )
                ),
                ###########################################UPLOAD ##########################################     
                
                tabPanel("Upload",
                         sidebarPanel(
                           # Allows user to select data file from local file system.
                           fileInput('file1', 'Choose file to upload',
                                     accept = c(
                                       'text/csv',
                                       'text/comma-separated-values',
                                       'text/tab-separated-values',
                                       'text/plain',
                                       '.csv',
                                       '.tsv'
                                     )
                           ),
                           tags$hr(),
                           
                           
                           radioButtons('sep', 'Separator:',
                                        c(Comma=',',
                                          Semicolon=';',
                                          Tab='\t'),
                                        ','),
                           tags$hr(),
                           
                           radioButtons('dir', 'Variables are listed in:',
                                        c(Rows = 'R',
                                          Columns = 'C'),
                                        'R'),
                           
                           tags$hr(),
                           
                           numericInput(inputId = "observations",label = "Number of observations to view:",value = 10),
                           
                           
                           textInput('title', "Plot Title", ''),
                           
                           tags$style(type="text/css",
                                      ".shiny-output-error { visibility: hidden; }",
                                      ".shiny-output-error:before { visibility: hidden; }"
                           ),
                           width = 3
                         ),
                         mainPanel(
                           tabsetPanel(
                             tabPanel("Dataset ",
                                      h4("Table"), # -------------------------------------- SXOLIA -------------
                                      tableOutput("table_data"),
                                      
                                      
                             ),
                             tabPanel("Summary", verbatimTextOutput("summary")),
                             ################################BOXPLOT#####################################
                             tabPanel("Boxplot",
                                      uiOutput('outSelections'),
                                      tags$hr(),
                                      plotOutput('boxPlot'),
                                      downloadButton('download1', 'Download', class = "buttCol"),
                                      tags$hr(),
                                      fluidRow(
                                        column(4,
                                               
                                               radioButtons('customY', "Set Custom Y Range", c(No = 'NotCustom', Yes =
                                                                                                 'Custom'), 'NotCustom', inline = TRUE)
                                        ),
                                        column(7, offset = 1,
                                               conditionalPanel(
                                                 condition = "input.customY == 'Custom'",
                                                 uiOutput('yMaxSelector')
                                               )
                                               
                                        )
                                      ),
                                      conditionalPanel(
                                        condition = "input.customY == 'Custom'",
                                        plotOutput('boxPlotCustomY'),
                                        downloadButton('download2', 'Download', class = "buttCol")
                                      ),
                                      
                                      tags$head(tags$style(".buttCol{background-color:#faf0e6;} .buttCol{color: purple;}")),
                                      
                             )
                             #######################################################################
                             
                           ),
                         )
                ),tabPanel("About",
                           
                )
                
                
    )
    
  ),
  
  
  
  
  #----------------------------------------------- SERVER CODE ----------------------------------------
  
  server = function(input, output) {
    
    #####################################   DATASET'S INFO  ###############################################
    
    # Varrate a summary of the data
    output$summary0 <- renderPrint({
      dataset <- numeric_dataset
      summary(dataset)
    })
    output$summaryAfter <- renderPrint({
      dataset <- creditcard
      summary(dataset)
    })
    
    # Varrate an HTML table view of the data ----
    output$table0 <- renderTable({
      head(numeric_dataset, n = input$obs)
    })
    output$tableAfter <- renderTable({
      head(creditcard, n = input$obs)
    })
    
    
    
    
    ############################################  Decision Tree ###########################################
    
    
    output$treeplot <- renderPlot({
      
      if(input$prun == 'Prunning'){
        
        plot(prune.creditcard)
        text(prune.creditcard,font=4, pretty=0)
      }
      else {
        plot(tree.creditcard)
        text(tree.creditcard,font=4, pretty=0)
      }
    })
    
    
    acc <- reactive({
      
      if(input$prun == 'Prunning'){
        return(confusionMatrix(table_acc2))
      }
      else {
        return(confusionMatrix(table_acc))
        
      }
    })
    
    
    output$accur <- renderPrint(acc())
    
    
    
    
    ########################################## PLOTS VISUALIZE ###########################################
    
    pl1<-reactive({
      return(ggplot(data=numeric_dataset, aes_string(x = input$xcol)) + 
               geom_histogram( aes(fill=..count..)) +
               scale_fill_gradient("Count", low="lightgrey", high="black") )
    })
    
    output$plot1<-renderPlot({
      
      pl1()
      
    })
    
    output$download_hist <- downloadHandler(
      filename = 'downloadedPlot.png' ,
      content = function(file) {
        ggsave(file, plot = pl1(), device = "png")
      }
    )
    
    #-------barplot-----
    
    pl2<-reactive({  
      ggplot(creditcard, aes_string(x = input$xcol)) +
        geom_bar(aes_string(fill= input$ycol), position = position_stack(reverse = FALSE)) +
        theme(legend.position = "top") +  theme_bw()
      
    })
    
    output$plot2<-renderPlot({
      pl2()
      
    })
    
    
    output$download_BarPlot <-downloadHandler(
      filename = 'downloadedPlot.png' ,
      content = function(file) {
        ggsave(file, plot = pl2(), device = "png")
      }
    )
    
    
    
    #----- ScatterPlot -----
    
    pl3 <- reactive({  
      ggplot(creditcard, aes_string(input$xcol, input$ycol)) +
        geom_point(aes(colour = factor(dpnm))) +
        coord_cartesian( expand = FALSE) + theme_bw()
      
    })
    
    output$plot3 <- renderPlot({
      pl3()
      
    })
    
    
    output$download_ScatterPlot <- downloadHandler(
      filename = 'downloadedPlot.png' ,
      content = function(file) {
        ggsave(file, plot = pl3(), device = "png")
      }
      
    )
    
    
    
    
    ##########################################KNN ###################################################
    
    # Combine the selected variables into a new data frame
    selectedData2 <- reactive({
      numeric_dataset[, c(input$xcol1, input$ycol1)]
    })
    
    targetPoint <- reactive({
      selectedData2()[input$target,]
    })
    
    predictedClass <- reactive({
      knn(selectedData2(), targetPoint(), numeric_dataset$dpnm, input$k)
    })
    
    output$plot10 <- renderPlot({
      par(mar = c(5.1, 4.1, 0, 1))                
      plot(selectedData2(),
           col = unclass(numeric_dataset$dpnm),
           pch = 20, cex = 3,)                
      
      distances <- pdist(targetPoint(), selectedData2())
      nnIdxs <- order(as.matrix(distances))[1:input$k]                
      points(selectedData2()[nnIdxs,], pch = 3, cex = 4, lwd = 4, col="orange")
      
      points(selectedData2()[input$target,], pch = 4, cex = 4, lwd = 4, col=unclass(predictedClass()))
      
    })
    
    output$origClass <- reactive({numeric_dataset$dpnm[input$target]})
    output$predClass <- reactive({predictedClass()})
    
    ##########################################KNN No2    ##########################################
    
    
    knn2<-reactive({
      knn.k <- knn(train=traink, test=testk, cl=traindpnm, k= input$kn)
      a<-confusionMatrix(table(knn.k ,testdpnm))
      return(a)
    })
    
    output$knn1 <- renderPrint(knn2())
    
    output$plotKNN<- renderPlot({
      plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level",  main = "Accuracy Plot")
    })
    
    
    
    ################################# Naive Bayes ################################# 
    
    NB <- reactive({
      model = train(x,y,'nb',trControl=trainControl(method='cv',number=10))
      Predict <- predict(model,newdata = testing )
      cf = confusionMatrix(Predict, testing$dpnm )
      return(cf)
    })
    
    output$NBayes <- renderPrint(NB())
    
    
    
    ################################ ASSOCIATION RULES  ################################
    
    
    rules <- reactive({
      if(input$rhs == 'No_rhsdpnm'){
        income_rules <- apriori(apriori_dataset, parameter=list (supp= as.numeric(input$sup),conf = as.numeric(input$conf) , minlen= as.numeric(input$len)+1, maxlen = as.numeric(input$mlen),
                                                                 maxtime=as.numeric(input$time), target = "rules") )
      }
      else {
        income_rules <- apriori(apriori_dataset, parameter=list (supp= as.numeric(input$sup),conf = as.numeric(input$conf) , minlen= as.numeric(input$len)+1, maxlen = as.numeric(input$mlen),
                                                                 maxtime=as.numeric(input$time), target = "rules"),appearance = list (rhs=c("dpnm=YES", "dpnm=NO")) )
        
      }
      
    })
    
    output$scatterPlot <- renderPlot({
      ar <- rules()
      plot(ar, method='scatterplot')
    }, height=600, width=600)
    
    
    
    output$graphPlot <- renderPlot({
      ar <- rules()
      plot(ar, method="graph")
    }, height=600, width=600 )
    
    # Grouped matrix plot
    output$groupedMatrix <- renderPlot({
      ar <- rules()
      plot(ar, method="grouped")
    }, height=600, width=600) 
    
    
    
    output$rules <- DT::renderDataTable( {
      
      irules <- rules()
      
      rulesdf <- DATAFRAME(irules)
      rulesdf
      #inspect(head(sort(irules, by='confidence'),10))
      
      
    })
    
    output$downloadData <- downloadHandler(
      filename = 'arules_data.csv',
      content = function(file) {
        write.csv(rules(), file)
      }
    )
    
    output$FreqPlot <- renderPlot({
      ar <- rules()
      itemFrequencyPlot(items(ar), topN = 10)
    })
    
    
    ##############################BOXPLOT##############################
    
    
    
    ####################Get Upload File############
    
    #Function to read data from a file, after checking file in not NULL.
    getData <- function(dataFile){
      inFile <- dataFile
      if (is.null(inFile)) {
        return(NULL)
      }
      else {
        read.csv(inFile$datapath, header = TRUE, sep = input$sep, quote = input$quote, row.names = 1,
                 stringsAsFactors = FALSE)
      }
    }
    
    # Read in the data from the input file.
    myData <- reactive({getData(input$file1)})
    
    dataDimensions <- reactive(dim(myData()))
    
    
    output$table_data <- renderTable({
      head(myData(),n = input$observations)
      
    })
    
    
    # Varrate a summary of the data for Plots
    output$summary <- renderPrint({
      dataset <- myData()
      summary(dataset)
    })
    
    
    
    #------------------------ KANEI TO FORMAT -------------------------
    formattedData <- function(){
      
      dimensions <- dim(myData())
      numRows <- dimensions[1]
      numCols <- dimensions[2]
      preformatted <- myData()
      # reformat
      preformatted <- preformatted[1:numRows, 1:numCols]
      preformatted <- lapply(preformatted, as.numeric)
      preformatted <- as.data.frame(preformatted)
      
      if(input$dir == 'C'){
        preformatted <- t(preformatted)
      }
      return(preformatted)
    }
    #-------------------------------------------------------------------------------
    
    
    # Get row.names for drop down menu from file read in to the app.
    output$outSelections <- renderUI({
      variables <- row.names(formattedData())
      selectInput("variables", "Please click the box below for drop-down menu to select variables to plot:",
                  choices = variables, multiple=TRUE, selectize=TRUE)
      
    })
    
    # Get maximum count value in data to use as y-max for slider input widget.
    output$yMaxSelector <- renderUI({
      
      maxForm <- round(max(formattedData()), 0)
      sliderInput('yRange', "To change range of y-axis, enter a new maximum value", 0, maxForm,
                  value = maxForm)
      
    })
    
    
    
    
    
    #--------------------- Sunartiseis gia dimiourgia boxplots -------------
    # BOXPLOT
    boxPlotInput <- function(){
      dataForPlot <- formattedData()
      dataForPlot <- t(dataForPlot)
      dataToPlot <- dataForPlot[,input$variables]
      boxplot(dataToPlot, col="lightpink",  main=input$title, xlab="variables", ylab ="Counts")
    }
    
    # BOXPLOT ME DIKO MAS YRANGE
    boxPlotCustomYInput <- function(){
      dataForPlot <- formattedData()
      dataForPlot <- t(dataForPlot)
      dataToPlot <- dataForPlot[,input$variables]
      boxplot(dataToPlot, col="lightblue", main=input$title, xlab="variables", ylab =
                "counts", ylim = c(0, input$yRange))
    }
    
    #OUTPUTS: kalei tis sunartiseis 
    output$boxPlot <- renderPlot({
      boxPlotInput()
    })
    
    output$boxPlotCustomY <- renderPlot({
      boxPlotCustomYInput()
    })
    
    
    #-------------------DOWNLOAD PLOTS----------------------------------------------------------------------
    
    # Download boxplot (no scaling)
    output$download1 <- downloadHandler(
      filename = 'downloadedPlot.png',
      content = function(file) {
        png(file)
        boxPlotInput()
        dev.off()
      }
    )
    
    # Download boxplot with scaling.
    output$download2 <- downloadHandler(
      filename = 'downloadedPlot.pdf',
      content = function(file) {
        pdf(file)
        boxPlotCustomYInput()
        dev.off()
        
      }
    )
    
    
    
    
    
    
  }
)