import torch
import torch.nn as nn
import torch.nn.functional as F
if_lora_grad=True
if_svd=True
if_sparse=True
import torch
import torch.optim as optim
#---------STE------------
class SignSTE_BirealFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Initialize gradient to zero
        grad_input = torch.zeros_like(input)
        
        # Define masks for different intervals
        mask1 = input < -1
        mask2 = (input >= -1) & (input < 0)
        mask3 = (input >= 0) & (input < 1)
        mask4 = input >= 1
        
        # Compute gradients for different intervals
        grad_input += (2 * input + 2) * mask2.type_as(input)
        grad_input += (-2 * input + 2) * mask3.type_as(input)
        # For mask1 and mask4, gradients remain zero
        
        # Multiply custom gradient with upstream gradient
        return grad_output * grad_input

class SignSTE_Bireal(nn.Module):
    def forward(self, input):
        return SignSTE_BirealFunc.apply(input)

#---------unishortcut------------
class ChannelResidualConnectionforconv(nn.Module):
    def __init__(self, in_channels, out_channels,scale_factor=None):
        super(ChannelResidualConnectionforconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if scale_factor is not None:
            self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        else:
            self.scale_factor = nn.Parameter(torch.ones(1, self.out_channels,1,1), requires_grad=True)

    def forward(self, x):
        x=x.to(self.scale_factor.device)
        batch_size, in_channels, height, width = x.size()

        if self.in_channels == self.out_channels:
            # If Cin == Cout, return input directly
            return x * self.scale_factor
        elif self.in_channels == self.out_channels * (self.in_channels // self.out_channels):
            # If Cin = n * Cout, group Cin channels by n, sum and average, corresponding to one output channel
            n = self.in_channels // self.out_channels
            x = x.view(batch_size, self.out_channels, n, height, width)
            x = x.mean(dim=2)
            return x * self.scale_factor
        elif self.out_channels == self.in_channels * (self.out_channels // self.in_channels):
            # If Cout = n * Cin, repeat Cin n times along the Cin dimension, corresponding to n output channels
            n = self.out_channels // self.in_channels
            x = x.repeat(1, n, 1, 1)
            return x * self.scale_factor
        else:
            # If Cin and Cout are not divisible, use asymmetric channel padding
            if self.in_channels > self.out_channels:
                # If Cin > Cout, split Cin channels by Cout, sum and average, corresponding to one output channel
                res = self.in_channels % self.out_channels
                n = self.in_channels // self.out_channels
                x_mod = x[:, self.in_channels-res:, :, :]
                x = x[:, :self.in_channels-res, :, :]
                x = x.view(batch_size, self.out_channels, n, height, width)
                x = x.mean(dim=2)
                x[:, :res, :, :]=(x[:, :res, :, :]*n + 1 * x_mod)/(n+1)
                return x* self.scale_factor
            else:
                # If Cout > Cin, repeat Cin multiple times along the Cin dimension, corresponding to Cout output channels
                res = self.out_channels % self.in_channels
                n = self.out_channels // self.in_channels
                x_plus = x.repeat(1, n+1, 1, 1)
                x_mod = x_plus[:, :n*self.in_channels+res, :, :]
                return x_mod* self.scale_factor
    def get_equiv_weight(self):
        testinput=torch.eye(self.in_channels).unsqueeze(2).unsqueeze(3)
        return self(testinput).squeeze(3).squeeze(2).t().unsqueeze(2).unsqueeze(3)

class ChannelResidualConnectionforlinear(nn.Module):
    def __init__(self, in_features, out_features,scale_factor=None):
        super(ChannelResidualConnectionforlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if scale_factor is not None:
            self.scale_factor = nn.Parameter(scale_factor, requires_grad=True)
        else:
            self.scale_factor = nn.Parameter(torch.ones(1, self.out_features), requires_grad=True)

        # Ensure in_features and out_features are divisible
        assert self.in_features % self.out_features == 0 or self.out_features % self.in_features == 0, \
            "in_features and out_features should be divisible."

    def forward(self, x):
        x=x.to(self.scale_factor.device)
        if x.dim() == 2:  # 2D input (batch_size, in_features)
            batch_size, in_features = x.size()

            if self.in_features == self.out_features:
                # If input and output feature dimensions are the same, return input directly
                return x * self.scale_factor

            elif self.in_features > self.out_features:
                # If in_features > out_features, perform averaging
                n = self.in_features // self.out_features
                x = x.view(batch_size, self.out_features, n)  # Reshape tensor
                x = x.mean(dim=2)  # Average each group
                return x * self.scale_factor

            elif self.out_features > self.in_features:
                # If out_features > in_features, perform repetition
                n = self.out_features // self.in_features
                x = x.repeat(1, n)  # Repeat along feature dimension
                return x * self.scale_factor
        
        elif x.dim() == 3:  # 3D input (batch_size, in_features, length)
            batch_size, in_features, length = x.size()

            if self.in_features == self.out_features:
                # If input and output feature dimensions are the same, return input directly
                return x * self.scale_factor

            elif self.in_features > self.out_features:
                # If in_features > out_features, perform averaging
                n = self.in_features // self.out_features
                x = x.view(batch_size, length, n, self.out_features)  # Reshape tensor
                x = x.mean(dim=2)  # Average each group, keep batch and length dimensions
                return x * self.scale_factor

            elif self.out_features > self.in_features:
                # If out_features > in_features, perform repetition
                n = self.out_features // self.in_features
                x = x.repeat(1, 1,n)  # Repeat along feature dimension, keep length dimension
                return x * self.scale_factor

        else:
            raise ValueError("Input tensor must be either 2D or 3D.")
    def get_equiv_weight(self):
        testinput=torch.eye(self.in_features)
        return self(testinput).t()
#----------sparse_skip-----------------------
def extract_topk_sparse_matrix(A: torch.Tensor, n: int) -> torch.Tensor:
    """
    Extract the top-n absolute values from matrix A and return as indices

    Parameters:
    - A (torch.Tensor): Input 2D matrix.
    - n (int): Number of non-zero elements to extract.

    Returns:
    - index ,value: Sparse matrix containing the top-n absolute values from A.
    """
    # Validate input matrix is 2D
    if A.dim() != 2:
        raise ValueError("Input matrix A must be 2D.")

    # Get matrix shape
    rows, cols = A.shape

    # Calculate total number of elements in the matrix
    total_elements = rows * cols

    # Validate n
    if n <= 0:
        raise ValueError("Parameter n must be a positive integer.")
    if n > total_elements:
        raise ValueError(f"Parameter n cannot exceed the total number of elements in matrix A {total_elements}.")

    # Flatten matrix A to 1D
    A_flat = A.view(-1)

    # Calculate absolute values of each element
    A_abs = torch.abs(A_flat)

    # Use torch.topk to extract the top-n absolute values and their indices
    topk_values, topk_indices = torch.topk(A_abs, n, largest=True, sorted=False)

    # Get original values (preserve sign)
    topk_values_signed = A_flat[topk_indices]

    # Convert 1D indices to 2D coordinates
    row_indices = topk_indices // cols
    col_indices = topk_indices % cols

    # Stack row and column indices to form a [2, n] index tensor
    indices = torch.stack([row_indices, col_indices], dim=0)

    return indices,topk_values_signed

class SparseMatrixLayer(nn.Module):
    def __init__(self, indices: torch.Tensor, values: torch.Tensor, shape: tuple, device='cpu'):
        """
        Initialize sparse matrix layer.

        Parameters:
        - indices (torch.Tensor): Indices of the sparse matrix, shape [2, nnz]. Stored as float for EMA.
        - values (torch.Tensor): Non-zero values of the sparse matrix, shape [nnz].
        - shape (tuple): Shape of the sparse matrix (rows, cols).
        - device (str or torch.device): Device type ('cpu' or 'cuda').
        """
        super(SparseMatrixLayer, self).__init__()
        
        # Input validation
        if indices.dim() != 2 or indices.size(0) != 2:
            raise ValueError("indices must be a 2D tensor with shape [2, nnz].")
        if values.dim() != 1:
            raise ValueError("values must be a 1D tensor.")
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("shape must be a tuple with two elements (rows, cols).")
        if indices.size(1) != values.size(0):
            raise ValueError("Number of columns in indices must match the length of values.")
        
        self.cin, self.cout = shape  # Input and output channels
        self.register_buffer('indices_float', indices.to(device).float())  # Store as float
        # Set values as trainable parameters
        self.values = nn.Parameter(values.to(device))
        self.shape = (self.cout, self.cin)  # Shape of the sparse matrix [out_features, in_features]

    def forward(self, x):
        """
        Forward pass, perform sparse matrix multiplication.

        Parameters:
        - x (torch.Tensor): Input activations, can be 2D, 3D, or 4D tensor. Suitable for convolutional, projection, and linear layers.

        Returns:
        - torch.Tensor: Result of sparse matrix multiplication with input activations.
        """
        device = x.device
        # Convert indices from float to long using rounding
        indices_long = torch.round(self.indices_float).long()
        # Create sparse matrix, ensure it is on the same device as the input
        sparse_matrix = torch.sparse_coo_tensor(
            indices_long, self.values, size=self.shape, device=device
        ).coalesce()

        if x.dim() == 4:
            # Input shape: [B, in_features, H, W]
            B, in_features, H, W = x.shape
            if in_features != self.cin:
                raise ValueError(f"Input in_features ({in_features}) does not match the number of rows in the sparse matrix ({self.cin}).")
            # Reshape to [in_features, B * H * W]
            x_reshaped = x.permute(1, 0, 2, 3).reshape(in_features, -1)  # [in_features, B*H*W]
            # Sparse matrix multiplication with x_reshaped: [out_features, B*H*W]
            out = torch.sparse.mm(sparse_matrix, x_reshaped)  # [out_features, B*H*W]
            # Reshape back to [out_features, B, H, W] and permute back to [B, out_features, H, W]
            out = out.reshape(self.cout, B, H, W).permute(1, 0, 2, 3)  # [B, out_features, H, W]
            return out

        elif x.dim() == 3:
            # Input shape: [B, length, in_features]
            B, length, in_features = x.shape
            if in_features != self.cin:
                raise ValueError(f"Input in_features ({in_features}) does not match the number of rows in the sparse matrix ({self.cin}).")
            # Reshape to [in_features, B * length]
            x_reshaped = x.permute(2, 0, 1).reshape(in_features, -1)  # [in_features, B*length]
            # Sparse matrix multiplication with x_reshaped: [out_features, B*length]
            out = torch.sparse.mm(sparse_matrix, x_reshaped)  # [out_features, B*length]
            # Reshape back to [out_features, B, length] and permute back to [B, length, out_features]
            out = out.reshape(self.cout, B, length).permute(1, 2, 0)  # [B, length, out_features]
            return out

        elif x.dim() == 2:
            # Input shape: [B, in_features]
            B, in_features = x.shape
            if in_features != self.cin:
                raise ValueError(f"Input in_features ({in_features}) does not match the number of rows in the sparse matrix ({self.cin}).")
            # Sparse matrix multiplication with x_reshaped: [out_features, B]
            out = torch.sparse.mm(sparse_matrix, x.t())  # [out_features, B]
            # Reshape back to [B, out_features]
            out = out.t()  # [B, out_features]
            return out

        else:
            raise ValueError("Input tensor must be 2D, 3D, or 4D.")

    def get_equiv_weight(self):
        """
        Return the equivalent dense matrix of the sparse matrix.

        Returns:
        - torch.Tensor: Dense matrix, shape (cout, cin).
        """
        # Convert indices_float to long via rounding
        indices_long = torch.round(self.indices_float).long()
        dense_matrix = torch.sparse_coo_tensor(
            indices_long, self.values, size=self.shape, device=self.values.device
        ).to_dense()
        return dense_matrix
#----------lora-------------
class LoRA1x1Conv(nn.Module):
    def __init__(self, in_channels, out_channels, rank, pretrained_weight=None):
        super(LoRA1x1Conv, self).__init__()

        self.rank = rank
        
        # Check if pretrained weight is provided
        if pretrained_weight is not None:
            # Check dimensions of pretrained weight
            if pretrained_weight.ndimension() == 4 and pretrained_weight.shape[2] == pretrained_weight.shape[3]:
                # Pretrained weight shape should be [out_channels, in_channels, 1, 1]
                weight_avg = pretrained_weight.mean(dim=[2, 3])  # [out_channels, in_channels]
                weight_matrix = weight_avg.t()  # [in_channels, out_channels]
                
                # SVD decomposition of weight matrix
                U, S, V = torch.svd(weight_matrix)
                
                # Initialize A and B with the first rank singular values
                self.A = nn.Conv2d(in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
                self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

                with torch.no_grad():
                    # Initialize weights using the square root of singular values
                    self.A.weight.data = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))  # [in_channels, rank]
                    self.A.weight.data = self.A.weight.data.t().unsqueeze(2).unsqueeze(3)  # [in_channels, rank, 1, 1]
                    self.B.weight.data = torch.diag(torch.sqrt(S[:rank])) @ V[:, :rank].t()  # [rank, out_channels]
                    self.B.weight.data = self.B.weight.data.t().unsqueeze(2).unsqueeze(3)  # [rank, out_channels, 1, 1]
                if if_lora_grad == False:
                # For conv layer
                    self.A.weight.requires_grad = False
                    self.B.weight.requires_grad = False


            else:
                raise ValueError("Unsupported weight matrix dimensions. LoRA1x1Conv expects 4D convolutional weights with square kernels.")
        else:
            self.A = nn.Conv2d(in_channels, rank, kernel_size=1, stride=1, padding=0, bias=False)
            # Default initialization
            nn.init.normal_(self.A.weight, mean=0.0, std=1e-4)
            self.B = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # x: [B, in_channels, H, W]
   #     print(x.size(), self.A.weight.size())
        intermediate = self.A(x)  # [B, rank, H, W]
   #     print(intermediate.size(), self.B.weight.size())
        out = self.B(intermediate)  # [B, out_channels, H, W]
        return out

    def get_equiv_weight(self):
        # Get equivalent weight matrix A @ B
    #    print(self.A.weight.size(), self.B.weight.size())
        equiv_weight = torch.matmul(self.A.weight.squeeze(3).squeeze(2).t(),
                                   self.B.weight.squeeze(3).squeeze(2).t())  # [out_channels, in_channels]
        equiv_weight = equiv_weight.t().unsqueeze(2).unsqueeze(3)  # [out_channels, in_channels, 1, 1]
        return equiv_weight

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, pretrained_weight=None):
        super(LoRALinear, self).__init__()

        # Ensure rank is within a reasonable range
        self.rank = min(rank, in_features, out_features)

        # Define low-rank linear layers A and B
        self.B = nn.Linear(in_features, self.rank, bias=False)  # [B: in_features -> rank]
        self.A = nn.Linear(self.rank, out_features, bias=False)  # [A: rank -> out_features]

        if pretrained_weight is not None:
            # Check dimensions of pretrained weight
            if pretrained_weight.ndimension() == 2 and pretrained_weight.shape == (out_features, in_features):
                # SVD decomposition of weight matrix
                U, S, V = torch.svd(pretrained_weight)
                # Initialize A and B with the first rank singular values
                with torch.no_grad():
                    self.A.weight.data = U[:, :self.rank] @ torch.diag(torch.sqrt(S[:self.rank]))  # [out_features, rank]
                    self.B.weight.data = (torch.diag(torch.sqrt(S[:self.rank])) @ V[:, :self.rank].t())  # [rank, in_features]
            else:
                raise ValueError("Pretrained weight dimensions do not match. Expected (out_features, in_features) matrix.")
            if if_lora_grad == False:
                self.A.weight.requires_grad = False
                self.B.weight.requires_grad = False
        else:
            # Default initialization
            nn.init.normal_(self.A.weight, mean=0.0, std=1e-4)

    def forward(self, x):
        # LoRA low-rank adaptation
        out = self.A(self.B(x))  # Pass through B, then A, low-rank adaptation
        return out

    def get_equiv_weight(self):
        # Return equivalent weight matrix A @ B
        #print(self.A.weight.size(), self.B.weight.size())
        return torch.matmul(self.A.weight, self.B.weight)

#---------HORQ------------
def HORQ_grad(real_weights, N=1):
    residual = real_weights.clone()
    binary_total = torch.zeros_like(real_weights)

    # Determine if it is a convolutional or linear layer based on weight shape
    if real_weights.ndimension() == 4:
        # Convolutional layer
        for i in range(N):
            # Compute scaling factor (average absolute value per channel)
            scaling_factor = torch.mean(residual.abs(), dim=[ 1,2, 3], keepdim=True).detach()
            # Construct binary approximation based on current residual sign and scaling factor
            binary_approx = scaling_factor * torch.sign(residual)
            
            # Accumulate binary approximation
            binary_total += binary_approx
            # Update residual
            residual = residual - binary_approx

    elif real_weights.ndimension() == 2:
        # Linear layer
        for i in range(N):
            # Compute scaling factor (average absolute value per feature)
            scaling_factor = torch.mean(residual.abs(), dim=1, keepdim=True).detach()
            # Construct binary approximation based on current residual sign and scaling factor
            binary_approx = scaling_factor * torch.sign(residual)
            
            # Accumulate binary approximation
            binary_total += binary_approx
            # Update residual
            residual = residual - binary_approx

    else:
        raise ValueError("Unsupported weight shape, only supports convolutional (4D) and linear (2D) layers")

    # Process binary total and return
    binary_total = binary_total.detach() - torch.clamp(real_weights, -1.0, 1.0).detach() + torch.clamp(real_weights, -1.0, 1.0)
    return binary_total

def HORQ_grad_activation(activations, N=1, oursign=SignSTE_Bireal()):
    """
    Binarize activations.

    Parameters:
        activations (torch.Tensor): Original activations, supports 2D (linear layer), 3D (embedding layer), and 4D (convolutional layer) tensors.
        N (int): Number of binarization iterations, default is 1.

    Returns:
        torch.Tensor: Binarized activations.
    """
    residual = activations.clone()
    binary_total = torch.zeros_like(activations)

    if activations.ndimension() == 4:
        # Convolutional layer activations [batch_size, channels, height, width]
        for _ in range(N):
            # Compute scaling factor (average absolute value across batch and spatial dimensions)
            scaling_factor = torch.mean(residual.abs(), dim=[1], keepdim=True).detach()
            # Construct binary approximation based on current residual sign and scaling factor
            binary_approx = scaling_factor * oursign(residual)
            
            # Accumulate binary approximation
            binary_total += binary_approx
            # Update residual
            residual = residual - binary_approx

    elif activations.ndimension() == 3:
        # Embedding layer activations [batch_size, seq_length, embedding_dim]
        for _ in range(N):
            # Compute scaling factor (average absolute value across batch and sequence length)
            scaling_factor = torch.mean(residual.abs(), dim=[2], keepdim=True).detach()
            # Construct binary approximation based on current residual sign and scaling factor
            binary_approx = scaling_factor * oursign(residual)
            
            # Accumulate binary approximation
            binary_total += binary_approx
            # Update residual
            residual = residual - binary_approx

    elif activations.ndimension() == 2:
        # Linear layer activations [batch_size, features]
        for _ in range(N):
            # Compute scaling factor (average absolute value across batch)
            scaling_factor = torch.mean(residual.abs(), dim=1, keepdim=True).detach()
            # Construct binary approximation based on current residual sign and scaling factor
            binary_approx = scaling_factor * oursign(residual)
            # Accumulate binary approximation
            binary_total += binary_approx
            # Update residual
            residual = residual - binary_approx

    else:
        raise ValueError("Unsupported activation shape, only supports convolutional (4D), embedding (3D), and linear (2D) layers")
    return binary_total

#--------CONV2D------------
class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
   #     print(x.size(),'rprelu')
        if x.ndimension() == 3:
        # If input is 3D tensor [batch_size, seq_length, features], adjust to [batch_size, features, seq_length]
            x = self.pr_bias0(x)
            x = x.permute(0, 2, 1) 
            x = self.pr_prelu(x).permute(0, 2, 1)  # Restore original dimension order
            x = self.pr_bias1(x)
            return x
        else :
            x = self.pr_bias0(x)
            x = self.pr_prelu(x)
            x= self.pr_bias1(x)
            return x

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        # Bias only needs to match the number of channels, other dimensions are broadcasted
        self.bias = nn.Parameter(torch.zeros(1, out_chn), requires_grad=True)

    def forward(self, x):
        if x.ndimension() == 2:
            # Input is 2D data [batch_size, features]
            bias2d=self.bias
            if bias2d.size(1) != x.size(1):
                raise ValueError("Bias size does not match input features size:1d")
            out = x + bias2d
        elif x.ndimension() == 3:
            bias3d=self.bias.unsqueeze(1)
            # Input is 3D data [batch_size, seq_length, features]
            if bias3d.size(2) != x.size(2):
                raise ValueError("Bias size does not match input channels size:2d")
            out = x + self.bias3d
        elif x.ndimension() == 4:
            bias4d=self.bias.unsqueeze(2).unsqueeze(3)
            # Input is 4D data [batch_size, channels, height, width]
            if bias4d.size(1) != x.size(1):
                raise ValueError("Bias size does not match input channels size:3d")
            out = x + bias4d
        else:
            raise ValueError("Unsupported input dimension. Only 2D, 3D, and 4D inputs are supported.")

        return out

class BinaryConv2d_ours(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=True,pretained_weight=None,rank=8,if_lora=True,if_skip=False):
        super(BinaryConv2d_ours, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.num=max(in_chn,out_chn)
        self.move0 = LearnableBias(self.in_channels)
        self.relu = RPReLU(self.out_channels)
        self.oursign = SignSTE_Bireal()#SignSTE_Tanh()
        self.rank = rank#LoRA rank
        if pretained_weight is not None:
            self.weight = pretained_weight
        self.if_lora = if_lora
        self.if_skip = if_skip
        if self.if_skip :
            pad = kernel_size//2
            if if_sparse:
                index,value=extract_topk_sparse_matrix(self.weight.data.mean(dim=[2,3]),self.num*2)
                self.skip_connection = SparseMatrixLayer(index, value, (in_chn, out_chn)).to(self.weight.device)
                self.weight=torch.nn.Parameter(self.weight- F.pad(self.skip_connection.get_equiv_weight().unsqueeze(2).unsqueeze(3), (pad, pad, pad, pad)))
            else :
                self.skip_connection = ChannelResidualConnectionforconv(in_channels=in_chn, out_channels=out_chn).to(self.weight.device)

        if self.if_lora : 
            pad = kernel_size//2
            if if_svd:
                self.lora=LoRA1x1Conv(in_channels=in_chn, out_channels=out_chn, rank=self.rank,pretrained_weight=self.weight).to(self.weight.device)
            else:
                self.lora=LoRA1x1Conv(in_channels=in_chn, out_channels=out_chn, rank=self.rank).to(self.weight.device)
            self.weight=torch.nn.Parameter(self.weight- F.pad(self.lora.get_equiv_weight(), (pad, pad, pad, pad)))

        self.order_activation = 1#HORQ order
        self.order_weight = 1
        self.weight_quantize = HORQ_grad
        self.activation_quantize=HORQ_grad_activation
        self.cin=in_chn
        self.cout=out_chn
        self.kernel_size=kernel_size
    def forward(self, x):
        x_raw = x
        #------------Activation quantization--------------
       # x = self.move0(x)
      #  x = self.oursign(x)
        x = self.activation_quantize(x,N=self.order_activation,oursign=self.oursign)
        #-----------Weight quantization and convolution--------------
        real_weights = self.weight
        binary_weights = self.weight_quantize(real_weights, N=self.order_weight)
        x = F.conv2d(x, binary_weights, self.bias, stride=self.stride, padding=self.padding)
        x = self.relu(x)
        #------------lora and skip connection----------------------
        if self.if_skip:

            if self.in_channels == self.out_channels:
                if x_raw.shape[2] < x.shape[2]:
                    shortcut = F.interpolate(x_raw, scale_factor=2, mode="nearest")
                elif x_raw.shape[2] > x.shape[2]:
                    shortcut = F.avg_pool2d(x_raw, kernel_size=self.stride, stride=self.stride)
                else:
                    shortcut = self.skip_connection(x_raw)
                x=x+shortcut
            else:
                shortcut = self.skip_connection(x_raw)
                x=x+shortcut
        if self.if_lora and x_raw.shape[2] == x.shape[2]:
            x=x+self.lora(x_raw)
        return x

def init_BinaryConv2d_ours_from_conv_with_lora(conv):
    binary_conv = BinaryConv2d_ours(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None,conv.weight,if_lora=True,if_skip=False)
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv

def init_BinaryConv2d_ours_from_conv_with_skip_and_lora(conv,rank=8,if_LRMB=True,if_SMB=True):
    binary_conv = BinaryConv2d_ours(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None,conv.weight,rank=rank,if_lora=if_LRMB,if_skip=if_SMB)
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv

def init_BinaryConv2d_ours_from_conv_with_skip(conv):
    binary_conv = BinaryConv2d_ours(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None,conv.weight,if_lora=False,if_skip=True)
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv

def init_BinaryConv2d_ours_from_conv_with_nothing(conv):
    binary_conv = BinaryConv2d_ours(conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.bias is not None,conv.weight,if_lora=False,if_skip=False)
    if conv.bias is not None:
        binary_conv.bias = conv.bias
    return binary_conv

#--------LINEAR------------
class BinaryLinear_ours(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,pretained_weight=None,rank=8,if_lora=True,if_skip=False):
        super(BinaryLinear_ours, self).__init__(in_features, out_features, bias=bias)
        if pretained_weight is not None:
            self.weight = pretained_weight
        self.num=max(in_features,out_features)
        self.channel_threshold = torch.nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
        self.rank =   rank
        self.if_lora = if_lora
        self.if_skip = if_skip
        if self.if_skip:
            if if_sparse:
                index,value=extract_topk_sparse_matrix(self.weight.data,self.num*2)
                self.skip_connection = SparseMatrixLayer(index, value, (in_features, out_features)).to(self.weight.device)
                self.weight = nn.Parameter(self.weight.data-self.skip_connection.get_equiv_weight())  
            else:
                self.skip_connection = ChannelResidualConnectionforlinear(in_features, out_features).to(self.weight.device)
      
        if self.if_lora :
            if if_svd:
                self.lora=LoRALinear(in_features, out_features, rank=self.rank,pretrained_weight=self.weight).to(self.weight.device)
            else:
                self.lora=LoRALinear(in_features, out_features, rank=self.rank).to(self.weight.device)
            self.weight = nn.Parameter(self.weight.data-self.lora.get_equiv_weight())

        self.num=max(in_features,out_features)
        self.order_activation = 1
        self.order_weight = 1
        self.oursign = SignSTE_Bireal()
        self.weight_quantize = HORQ_grad
        self.activation_quantize=HORQ_grad_activation
        self.cin=in_features
        self.cout=out_features
    def forward(self, input):
        #------------lora and skip--------------
        if self.if_lora :
            lora=self.lora(input)
        if self.if_skip:
        
            shortcut = self.skip_connection(input)
        #------------Activation quantization--------------
       # x = input + self.channel_threshold
       # x = self.oursign(x) 
        x = self.activation_quantize(input,N=self.order_activation,oursign=self.oursign)
        #-----------Weight quantization and convolution--------------
        real_weights = self.weight
        binary_weights = self.weight_quantize(real_weights, N=self.order_weight)
        output = F.linear(x, binary_weights, self.bias)
        #-----------Add the lora and skip----------------------
        if self.if_lora:
            output=output+lora
        if self.if_skip:
            output=output+shortcut
        return output

def init_BinaryLinear_ours_from_Linear_with_lora(linear):
    binary_linear = BinaryLinear_ours(linear.in_features, linear.out_features, linear.bias is not None,linear.weight,if_lora=True,if_skip=False)
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear

def init_BinaryLinear_ours_from_Linear_with_skip_and_lora(linear,rank=8,if_LRMB=True,if_SMB=True):
    binary_linear = BinaryLinear_ours(linear.in_features, linear.out_features, linear.bias is not None,linear.weight,rank=rank,if_lora=if_LRMB,if_skip=if_SMB)
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear

def init_BinaryLinear_ours_from_Linear_with_skip(linear):
    binary_linear = BinaryLinear_ours(linear.in_features, linear.out_features, linear.bias is not None,linear.weight,if_lora=False,if_skip=True)
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear

def init_BinaryLinear_ours_from_Linear_with_nothing(linear):
    binary_linear = BinaryLinear_ours(linear.in_features, linear.out_features, linear.bias is not None,linear.weight,if_lora=False,if_skip=False)
    if linear.bias is not None:
        binary_linear.bias = linear.bias
    return binary_linear
